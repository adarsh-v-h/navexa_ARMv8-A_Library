#pragma once

// =============================================================================
// armv8lib/virt/virt.hpp
// Virtualization & Hypervisor Support — Header Only
//
// Provides software abstractions for ARMv8-A virtualization:
//
//   VmContext      — guest VM register/state snapshot (for context switching)
//   Stage2Entry    — stage-2 MMU page table entry abstraction
//   VirtInterrupt  — virtual interrupt descriptor
//
// These are pure C++ abstractions — no actual EL2 instructions are emitted.
// Intended for use in simulators, hypervisor prototypes, and OS research.
//
// Namespace : armv8::virt
// Standard  : C++17
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <array>
#include <stdexcept>
#include <string>

namespace armv8 {
namespace virt {

// -----------------------------------------------------------------------------
// VmContext — snapshot of a guest VM's CPU state
//
// Mirrors the registers saved/restored during a VM context switch:
//   x0..x30  — general purpose registers
//   sp, pc   — stack pointer, program counter
//   pstate   — processor state (NZCV, EL, DAIF flags)
//   vmid     — VM identifier (used by stage-2 MMU)
// -----------------------------------------------------------------------------
struct VmContext {
    std::array<uint64_t, 31> gpr{};   // x0..x30
    uint64_t sp     = 0;              // stack pointer (EL1)
    uint64_t pc     = 0;              // program counter
    uint64_t pstate = 0;              // PSTATE register
    uint32_t vmid   = 0;              // VM ID (VTTBR_EL2.VMID)

    // Save current state (software simulation — zeroes all regs)
    void reset() {
        gpr.fill(0);
        sp = pc = pstate = 0;
    }

    // Check if context is initialised
    bool is_valid() const { return vmid != 0; }

    // Accessors for named registers
    uint64_t& x(std::size_t n) {
        if (n > 30) throw std::out_of_range("VmContext: register x0..x30 only");
        return gpr[n];
    }
    const uint64_t& x(std::size_t n) const {
        if (n > 30) throw std::out_of_range("VmContext: register x0..x30 only");
        return gpr[n];
    }
};

// -----------------------------------------------------------------------------
// Stage2MemAttr — memory attribute for a stage-2 page table entry
// -----------------------------------------------------------------------------
enum class Stage2MemAttr : uint8_t {
    Device_nGnRnE = 0x00,   // strongly ordered device
    Normal_NC     = 0x05,   // normal, non-cacheable
    Normal_WB     = 0x0F,   // normal, write-back cacheable (most RAM)
};

// -----------------------------------------------------------------------------
// Stage2Perm — access permissions for stage-2 entry
// -----------------------------------------------------------------------------
enum class Stage2Perm : uint8_t {
    None  = 0b00,
    Read  = 0b01,
    Write = 0b10,
    RW    = 0b11,
};

// -----------------------------------------------------------------------------
// Stage2Entry — a single stage-2 MMU page table entry
//
// Models a 4KB page mapping from IPA (intermediate physical address)
// to PA (physical address) with attributes and permissions.
// -----------------------------------------------------------------------------
struct Stage2Entry {
    uint64_t      ipa   = 0;                              // input address
    uint64_t      pa    = 0;                              // output address
    Stage2MemAttr attr  = Stage2MemAttr::Normal_WB;
    Stage2Perm    perm  = Stage2Perm::RW;
    bool          valid = false;
    bool          af    = false;                          // access flag

    Stage2Entry() = default;
    Stage2Entry(uint64_t ipa, uint64_t pa,
                Stage2MemAttr attr = Stage2MemAttr::Normal_WB,
                Stage2Perm    perm = Stage2Perm::RW)
        : ipa(ipa), pa(pa), attr(attr), perm(perm), valid(true), af(false) {}

    // Translate IPA → PA (returns pa if valid, throws if not)
    uint64_t translate(uint64_t input_ipa) const {
        if (!valid) throw std::runtime_error("Stage2Entry: entry not valid");
        if (input_ipa != ipa) throw std::runtime_error("Stage2Entry: IPA mismatch");
        return pa;
    }

    // Check if access type is permitted
    bool permits_read()  const { return valid && (static_cast<uint8_t>(perm) & 0b01); }
    bool permits_write() const { return valid && (static_cast<uint8_t>(perm) & 0b10); }
};

// -----------------------------------------------------------------------------
// VirtIntType — type of virtual interrupt
// -----------------------------------------------------------------------------
enum class VirtIntType : uint8_t {
    IRQ  = 0,   // normal interrupt
    FIQ  = 1,   // fast interrupt
    SError = 2, // system error
};

// -----------------------------------------------------------------------------
// VirtInterrupt — a virtual interrupt descriptor
//
// Models an interrupt injected into a guest VM via the GIC virtual interface.
// -----------------------------------------------------------------------------
struct VirtInterrupt {
    uint32_t    intid    = 0;                    // interrupt ID (INTID)
    VirtIntType type     = VirtIntType::IRQ;
    uint8_t     priority = 0;                    // 0 = highest
    bool        pending  = false;
    bool        enabled  = false;
    uint32_t    vmid     = 0;                    // target VM

    VirtInterrupt() = default;
    VirtInterrupt(uint32_t intid, VirtIntType type, uint8_t priority, uint32_t vmid)
        : intid(intid), type(type), priority(priority),
          pending(false), enabled(true), vmid(vmid) {}

    // Assert the interrupt (mark pending)
    void assert_int()  { if (enabled) pending = true; }

    // Deassert (clear pending)
    void deassert_int() { pending = false; }

    // Check if interrupt should be delivered
    bool should_deliver() const { return enabled && pending; }
};

// -----------------------------------------------------------------------------
// ContextSwitchResult — returned by simulate_context_switch()
// -----------------------------------------------------------------------------
struct ContextSwitchResult {
    bool     success      = false;
    uint32_t from_vmid    = 0;
    uint32_t to_vmid      = 0;
    uint64_t cycles_est   = 0;   // estimated cycle cost (software model)
};

// -----------------------------------------------------------------------------
// simulate_context_switch()
//
// Simulates saving `from` context and restoring `to` context.
// In real hardware this involves ERET, VTTBR_EL2 updates, TLB invalidation.
// Here we model the register save/restore in software.
//
// Returns a ContextSwitchResult describing what happened.
// -----------------------------------------------------------------------------
inline ContextSwitchResult simulate_context_switch(VmContext& from,
                                                    VmContext& to)
{
    if (!from.is_valid() || !to.is_valid()) {
        return { false, from.vmid, to.vmid, 0 };
    }

    // Software model: "save" from by doing nothing (state already in struct)
    // "restore" to by marking it as the active context
    // In real VHE: write VTTBR_EL2, ISB, TLBI VMALLS12E1IS, DSB

    ContextSwitchResult result;
    result.success    = true;
    result.from_vmid  = from.vmid;
    result.to_vmid    = to.vmid;
    result.cycles_est = 200;  // typical ARMv8 VM switch ~200 cycles

    return result;
}

} // namespace virt
} // namespace armv8

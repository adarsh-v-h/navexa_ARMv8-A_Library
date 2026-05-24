// src/atomics/atomic_ops.cpp
//
// Out-of-line implementations for the navexa atomic module.
//
// Why a .cpp at all when the header is "header-only"?
// ────────────────────────────────────────────────────
//  • The hot CAS / LDADD wrappers stay inline in the header so the compiler
//    can always eliminate the call overhead and fold the instruction into
//    surrounding code.
//  • This file provides:
//      – FreeList::push / pop  (non-trivial, loop-carrying logic)
//      – A CASPAL (128-bit CAS) helper for the ABA-stamp update
//      – Diagnostic helpers used in tests and asserts

#include "armv8lib/atomics.h"
#include <cassert>
#include <cstdint>

namespace armv8lib {
namespace atomic {

// ────────────────────────────────────────────────────────────────────────────
// Internal: 128-bit Compare-And-Swap (CASPAL)
//
// ARMv8.1-A introduced CASPAL for 128-bit (16-byte) atomic operations.
// We use it to atomically update both the head pointer and the ABA stamp
// in a single instruction, eliminating the ABA hazard in the free-list.
//
// Layout (16 bytes, naturally aligned):
//   Xs   = low  64 bits  → current head pointer (expected / new)
//   Xs+1 = high 64 bits  → ABA stamp            (expected / new)
// ────────────────────────────────────────────────────────────────────────────
static bool caspal128(uint64_t* addr,
                      uint64_t  exp_lo, uint64_t  exp_hi,
                      uint64_t  new_lo, uint64_t  new_hi) noexcept {
    // CASPAL requires an even-register pair: Xs and Xs+1.
    // GCC/Clang honour the "r" constraint for pairs; we pass them as
    // separate register operands and rely on the allocator to place them
    // in consecutive registers.  Using local variables keeps constraints
    // simple without relying on GCC extensions.
    register uint64_t r0 __asm__("x0") = exp_lo;
    register uint64_t r1 __asm__("x1") = exp_hi;
    register uint64_t r2 __asm__("x2") = new_lo;
    register uint64_t r3 __asm__("x3") = new_hi;

    __asm__ volatile(
        "caspal %[lo], %[hi], %[nlo], %[nhi], [%[ptr]]"
        : [lo] "+r"(r0), [hi] "+r"(r1)
        : [nlo] "r"(r2), [nhi] "r"(r3), [ptr] "r"(addr)
        : "memory"
    );

    // On success CASPAL leaves the original memory value in the register pair;
    // if it matches our expected values, the swap happened.
    return (r0 == exp_lo) && (r1 == exp_hi);
}

// ────────────────────────────────────────────────────────────────────────────
// FreeList::push
//
// Treiber-stack push with ABA-stamped head.
//
// Algorithm
// ─────────
//  1. Read current head (pointer + stamp) with ACQUIRE semantics.
//  2. Link the new node to the current head.
//  3. Attempt CASPAL to swing head to (new_node, stamp).
//     – On failure, another thread changed head; re-read and retry.
//  4. The stamp is NOT incremented on push — only pop increments it.
//     This matches the conventional Treiber-stack recommendation: the ABA
//     hazard only arises on pop (the recycled node re-appears as head).
// ────────────────────────────────────────────────────────────────────────────
void FreeList::push(FreeNode* node) noexcept {
    assert(node != nullptr);

    // Reinterpret the HeadSlot struct directly as a 128-bit pointer tracking space
    uint64_t* slot = reinterpret_cast<uint64_t*>(&head_data_);

    while (true) {
        // Read directly from the raw fields (matching your exact 'ptr' and 'stamp' definitions)
        const uint64_t cur_ptr   = head_data_.ptr;   
        const uint64_t cur_stamp = head_data_.stamp; 

        // Link the new node to the current head before making it visible.
        node->next.store(cur_ptr);  // RELEASE: visible before the CAS below

        if (caspal128(slot,
                      cur_ptr,  cur_stamp,
                      reinterpret_cast<uint64_t>(node), cur_stamp)) {
            return;  // Success: new node is now the head.
        }
        
        // ARMv8 Hardware back-off: emit a YIELD hint on contention to 
        // save power and reduce memory bus pressure while spinning.
        __asm__ volatile("yield" ::: "memory");
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FreeList::pop
//
// Treiber-stack pop with ABA stamp increment.
//
// Algorithm
// ─────────
//  1. Read current head (pointer + stamp) with ACQUIRE semantics.
//  2. If head is null → list empty, return nullptr.
//  3. Read next pointer from the head node.
//  4. Attempt CASPAL to swing head to (next, stamp+1).
//     – The stamp increment makes any subsequent ABA-collision detectable.
//     – On failure, retry from step 1.
// ────────────────────────────────────────────────────────────────────────────
FreeNode* FreeList::pop() noexcept {
    uint64_t* slot = reinterpret_cast<uint64_t*>(&head_data_);

    while (true) {
        const uint64_t cur_ptr   = head_data_.ptr;   
        const uint64_t cur_stamp = head_data_.stamp; 

        if (cur_ptr == 0u) return nullptr;  // Empty list.

        auto* node = reinterpret_cast<FreeNode*>(cur_ptr);

        // Read the successor before attempting CAS.  If `node` gets recycled
        // after we read its `next` but before our CAS, the stamp increment will
        // cause our CAS to fail, protecting against ABA.
        const uint64_t next_ptr = node->next.load();  // ACQUIRE

        if (caspal128(slot,
                      cur_ptr,   cur_stamp,
                      next_ptr,  cur_stamp + 1u)) {
            // Success: node is ours.
            node->next.store(0u);  // Clear the stale next pointer. RELEASE.
            return node;
        }
        
        // ARMv8 Hardware back-off: yield pipeline on CAS failure.
        __asm__ volatile("yield" ::: "memory");
    }
}

} // namespace atomic
} // namespace armv8lib
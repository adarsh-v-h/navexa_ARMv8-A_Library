#pragma once
// include/armv8lib/atomics.h
//
// ARMv8-A LSE (Large System Extensions) atomic primitives for navexa.

#ifndef ARMV8LIB_ATOMICS_H
#define ARMV8LIB_ATOMICS_H

#include <cstdint>
#include <cstddef>

#if !defined(__ARM_FEATURE_ATOMICS)
#  error "armv8lib/atomics.h requires ARMv8-A LSE. Compile with -march=armv8.5-a+lse"
#endif

namespace armv8lib {
namespace atomic {

// ════════════════════════════════════════════════════════════════════════════
// 1. Low-level LSE wrappers (64-bit)
// ════════════════════════════════════════════════════════════════════════════

[[nodiscard]] inline uint64_t
cas64_acq_rel(uint64_t* ptr, uint64_t expected, uint64_t desired) noexcept {
    __asm__ volatile(
        "casal %[old], %[new_val], [%[addr]]"
        : [old] "+r"(expected)
        : [new_val] "r"(desired), [addr] "r"(ptr)
        : "memory"
    );
    return expected;
}

[[nodiscard]] inline uint64_t
cas64_relaxed(uint64_t* ptr, uint64_t expected, uint64_t desired) noexcept {
    __asm__ volatile(
        "cas %[old], %[new_val], [%[addr]]"
        : [old] "+r"(expected)
        : [new_val] "r"(desired), [addr] "r"(ptr)
        : "memory"
    );
    return expected;
}

[[nodiscard]] inline uint64_t
ldadd64_acq_rel(uint64_t* ptr, uint64_t addend) noexcept {
    uint64_t result;
    __asm__ volatile(
        "ldaddal %[add], %[res], [%[addr]]"
        : [res] "=r"(result)
        : [add] "r"(addend), [addr] "r"(ptr)
        : "memory"
    );
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Typed wrappers
// ════════════════════════════════════════════════════════════════════════════

struct Atomic64 {
    explicit Atomic64(uint64_t initial = 0) noexcept : val_(initial) {}

    Atomic64(const Atomic64&)            = delete;
    Atomic64& operator=(const Atomic64&) = delete;

    [[nodiscard]] uint64_t load() const noexcept {
        uint64_t v;
        __asm__ volatile("ldar %[v], [%[addr]]" : [v] "=r"(v) : [addr] "r"(&val_) : "memory");
        return v;
    }

    void store(uint64_t desired) noexcept {
        __asm__ volatile("stlr %[v], [%[addr]]" : : [v] "r"(desired), [addr] "r"(&val_) : "memory");
    }

    [[nodiscard]] bool compare_exchange(uint64_t& expected, uint64_t desired) noexcept {
        const uint64_t prev = cas64_acq_rel(&val_, expected, desired);
        if (prev == expected) return true;
        expected = prev;
        return false;
    }

    [[nodiscard]] uint64_t fetch_add(uint64_t addend) noexcept {
        return ldadd64_acq_rel(&val_, addend);
    }

    [[nodiscard]] uint64_t fetch_sub(uint64_t subtrahend) noexcept {
        return ldadd64_acq_rel(&val_, -subtrahend);
    }

    uint64_t val_;
};

// ════════════════════════════════════════════════════════════════════════════
// 3. Atomic reference counter
// ════════════════════════════════════════════════════════════════════════════

class RefCount {
public:
    explicit RefCount(uint32_t initial = 1) noexcept : count_(initial) {}

    [[nodiscard]] uint32_t get() const noexcept {
        return count_;
    }

    void retain() noexcept {
        uint32_t old_val;
        uint32_t inc = 1u;
        // Replaced 'wzr' with an explicit increment register to properly add 1
        __asm__ volatile(
            "ldaddal %w[inc], %w[old], [%[addr]]"
            : [old] "=r"(old_val)
            : [inc] "r"(inc), [addr] "r"(&count_) 
            : "memory");
    }

    [[nodiscard]] bool release() noexcept {
        uint32_t old;
        __asm__ volatile(
            "ldaddal %w[neg], %w[old], [%[addr]]"
            : [old] "=r"(old)
            : [neg] "r"(static_cast<uint32_t>(-1)), [addr] "r"(&count_)
            : "memory");
        if (old == 1u) {
            __asm__ volatile("dmb ish" ::: "memory");
            return true;
        }
        return false;
    }

private:
    uint32_t count_;
};

// ════════════════════════════════════════════════════════════════════════════
// 4. Lock-free free-list
// ════════════════════════════════════════════════════════════════════════════

struct FreeNode;

class FreeList {
public:
    FreeList() noexcept = default;

    void push(FreeNode* node) noexcept;
    [[nodiscard]] FreeNode* pop() noexcept;

    [[nodiscard]] bool empty() const noexcept {
        return head_data_.ptr == 0u;
    }

private:
    // Required for CASPAL: contiguous 128-bit block, 16-byte aligned.
    struct alignas(16) HeadSlot {
        uint64_t ptr;
        uint64_t stamp;
    };

    static_assert(sizeof(HeadSlot) == 16, "HeadSlot must be 16 bytes for CASPAL");

    HeadSlot head_data_{0, 0};
    
    friend void push(FreeNode* node) noexcept;
    friend FreeNode* pop() noexcept;
    friend class atomic_ops; 
};

struct FreeNode {
    Atomic64 next{0};
    RefCount ref{1};
};

} // namespace atomic
} // namespace armv8lib

#endif // ARMV8LIB_ATOMICS_H
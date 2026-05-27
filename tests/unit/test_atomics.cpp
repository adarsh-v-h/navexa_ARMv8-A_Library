// tests/unit/test_atomics.cpp
//
// Stress-test suite for the navexa atomic module.
//
// Test strategy
// ─────────────
//  1. Single-threaded correctness – verify CAS / LDADD / RefCount semantics.
//  2. Multi-threaded stress      – hammer the FreeList from N threads to
//                                  expose races or ABA corruptions.
//  3. QEMU multi-core            – run with   qemu-aarch64 -smp 4
//                                  or via CTest's CROSSCOMPILING_EMULATOR.
//
// Build:
//   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64.cmake -B build
//   cmake --build build
//   ctest --test-dir build -R test_atomics -V
//
// Or directly:
//   qemu-aarch64 -smp 4 -L <sysroot> ./build/test_atomics

#include "armv8lib/atomics.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <array>

using namespace armv8lib::atomic;

// ── 1. Atomic64 basic operations ─────────────────────────────────────────

TEST(Atomic64, LoadStore) {
    Atomic64 a{42};
    EXPECT_EQ(a.load(), 42u);
    a.store(0xDEADBEEF);
    EXPECT_EQ(a.load(), 0xDEADBEEFu);
}

TEST(Atomic64, CompareExchangeSuccess) {
    Atomic64 a{10};
    uint64_t expected = 10;
    bool ok = a.compare_exchange(expected, 20);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.load(), 20u);
}

TEST(Atomic64, CompareExchangeFailure) {
    Atomic64 a{10};
    uint64_t expected = 99;           // wrong expected
    bool ok = a.compare_exchange(expected, 20);
    EXPECT_FALSE(ok);
    EXPECT_EQ(expected, 10u);         // expected updated to actual value
    EXPECT_EQ(a.load(), 10u);         // value unchanged
}

TEST(Atomic64, FetchAdd) {
    Atomic64 a{100};
    uint64_t old = a.fetch_add(5);
    EXPECT_EQ(old, 100u);
    EXPECT_EQ(a.load(), 105u);
}

TEST(Atomic64, FetchSub) {
    Atomic64 a{100};
    uint64_t old = a.fetch_sub(25);
    EXPECT_EQ(old, 100u);
    EXPECT_EQ(a.load(), 75u);
}

// ── 2. RefCount ──────────────────────────────────────────────────────────

TEST(RefCount, BasicLifecycle) {
    RefCount rc{1};
    EXPECT_EQ(rc.get(), 1u);

    rc.retain();
    EXPECT_EQ(rc.get(), 2u);

    bool last = rc.release();
    EXPECT_FALSE(last);
    EXPECT_EQ(rc.get(), 1u);

    last = rc.release();
    EXPECT_TRUE(last);
    EXPECT_EQ(rc.get(), 0u);
}

TEST(RefCount, MultipleRetains) {
    RefCount rc{1};
    for (int i = 0; i < 9; ++i) rc.retain();
    EXPECT_EQ(rc.get(), 10u);

    for (int i = 0; i < 9; ++i) EXPECT_FALSE(rc.release());
    EXPECT_TRUE(rc.release());
}

// ── 3. FreeList single-threaded ──────────────────────────────────────────

// Allocate a small pool of nodes on the stack for testing.
static std::array<FreeNode, 64> g_pool;

TEST(FreeList, EmptyPop) {
    FreeList fl;
    EXPECT_TRUE(fl.empty());
    EXPECT_EQ(fl.pop(), nullptr);
}

TEST(FreeList, PushPop) {
    FreeList fl;
    fl.push(&g_pool[0]);
    fl.push(&g_pool[1]);
    fl.push(&g_pool[2]);
    EXPECT_FALSE(fl.empty());

    // LIFO order expected.
    EXPECT_EQ(fl.pop(), &g_pool[2]);
    EXPECT_EQ(fl.pop(), &g_pool[1]);
    EXPECT_EQ(fl.pop(), &g_pool[0]);
    EXPECT_EQ(fl.pop(), nullptr);
    EXPECT_TRUE(fl.empty());
}

// ── 4. FreeList multi-threaded stress ────────────────────────────────────
//
// N_THREADS producer threads each push M_OPS nodes; N_THREADS consumer
// threads each pop until they have collected M_OPS nodes.  At the end every
// node must have been produced and consumed exactly once.

static constexpr int N_THREADS = 4;
static constexpr int M_OPS     = 1'000;

// Use a separate pool large enough for all threads.
static std::array<FreeNode, N_THREADS * M_OPS> g_stress_pool;

TEST(FreeList, MultiThreadedStress) {
    FreeList fl;

    // Reset nodes.
    for (auto& n : g_stress_pool) {
        n.next.store(0);
        new (&n.ref) RefCount{1};
    }

    std::atomic<int> total_popped{0};

    // Producer threads
    std::vector<std::thread> producers;
    producers.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        producers.emplace_back([&, t]() {
            for (int i = 0; i < M_OPS; ++i)
                fl.push(&g_stress_pool[t * M_OPS + i]);
        });
    }

    // Consumer threads
    std::vector<std::thread> consumers;
    consumers.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        consumers.emplace_back([&]() {
            int popped = 0;
            while (popped < M_OPS) {
                FreeNode* n = fl.pop();
                if (n) {
                    ++popped;
                    total_popped.fetch_add(1, std::memory_order_relaxed);
                } else {
                    // Busy-wait with a yield hint while producers are still running.
                    __asm__ volatile("yield");
                }
            }
        });
    }

    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    EXPECT_TRUE(fl.empty());
    EXPECT_EQ(total_popped.load(), N_THREADS * M_OPS);
}

// ── 5. RefCount multi-threaded stress ────────────────────────────────────
//
// Verifies that exactly one thread observes the transition to zero when
// N_THREADS threads each call release() once after a single retain burst.

TEST(RefCount, MultiThreadedRelease) {
    constexpr int NTHREADS = 8;
    RefCount rc{static_cast<uint32_t>(NTHREADS)};
    std::atomic<int> last_seen{0};

    std::vector<std::thread> threads;
    threads.reserve(NTHREADS);
    for (int i = 0; i < NTHREADS; ++i) {
        threads.emplace_back([&]() {
            if (rc.release())
                last_seen.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& t : threads) t.join();

    EXPECT_EQ(last_seen.load(), 1)
        << "Exactly one thread must observe the zero transition";
}
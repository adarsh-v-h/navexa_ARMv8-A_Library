/*
 * benchmarks/bench_sve.cpp
 * Micro-benchmark suite for the navexa SVE emulation layer and MathExt.
 *
 * Measures throughput (elements/second) and latency for every public
 * function in navexa::sve and navexa::math. Results are printed to stdout
 * in a human-readable table.
 *
 * Design:
 *   - Zero external dependencies: uses only <chrono> for timing.
 *   - Each benchmark runs for at least kMinTimeMs milliseconds so that
 *     short kernels are measured reliably.
 *   - Arrays are kept in L1/L2 cache (≤ 64 KiB) to measure compute
 *     throughput, not memory bandwidth.
 *   - Volatile sinks prevent the compiler from eliminating dead stores.
 *
 * Build (standalone, from repo root):
 *   aarch64-linux-gnu-g++ -O3 -march=armv8.5-a+memtag+lse \
 *       -I include \
 *       benchmarks/bench_sve.cpp \
 *       src/sve/sve_emulation.cpp \
 *       src/sve/neon_fallback.cpp \
 *       src/mathext/transcendentals.cpp \
 *       -o bench_sve
 *
 * Or via CMake if a benchmarks target is added to the root CMakeLists.
 *
 * Usage:
 *   ./bench_sve          — run all benchmarks
 *   ./bench_sve <filter> — run only benchmarks whose name contains <filter>
 *                          (case-sensitive substring match)
 */

#include "armv8lib/sve.h"
#include "armv8lib/mathext.h"

#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>

/* ── Benchmark configuration ─────────────────────────────────────────────── */

/* Minimum wall-clock time to run each benchmark (ms). */
static constexpr double kMinTimeMs = 200.0;

/* Array length for "medium" benchmarks (fits comfortably in L1/L2). */
static constexpr size_t kMedN = 4096;

/* Array length for "large" benchmarks (L2/L3 boundary test). */
static constexpr size_t kLargeN = 65536;

/* Warm-up iterations before timing begins. */
static constexpr int kWarmupIter = 3;

/* ── Utility: high-resolution timer ─────────────────────────────────────── */

using Clock = std::chrono::steady_clock;
using Ns    = std::chrono::nanoseconds;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               Clock::now().time_since_epoch()).count();
}

/* ── Volatile sink — prevents dead-code elimination of outputs ───────────── */

template <typename T>
static inline void do_not_optimize(const T* p, size_t n) {
    volatile const T* vp = p;
    (void)vp[0];
    (void)vp[n - 1];
}

static inline void do_not_optimize_f32(const float32x4_t v) {
    float buf[4];
    vst1q_f32(buf, v);
    volatile float x = buf[0];
    (void)x;
}

/* ── Result struct ───────────────────────────────────────────────────────── */

struct BenchResult {
    std::string name;
    double      mops;      /* million operations (elements) per second */
    double      ms_total;  /* total wall time of measured loop */
    long long   iters;     /* loop iterations completed */
};

/* ── Global results table ────────────────────────────────────────────────── */

static std::vector<BenchResult> g_results;
static std::string              g_filter;

/* ── Runner ──────────────────────────────────────────────────────────────── */

/*
 * run_bench(name, N, fn)
 *
 * Calls fn() in a loop until kMinTimeMs has elapsed (after a warm-up phase).
 * fn() must process exactly N elements each call.
 * Records throughput in millions of elements/second.
 */
static void run_bench(const std::string& name,
                      size_t             N,
                      std::function<void()> fn) {
    if (!g_filter.empty() && name.find(g_filter) == std::string::npos)
        return;

    /* Warm-up: run kWarmupIter times without timing */
    for (int w = 0; w < kWarmupIter; ++w) fn();

    /* Timed loop */
    long long iters    = 0;
    double    start_ms = now_ms();
    double    elapsed  = 0.0;

    do {
        fn();
        ++iters;
        elapsed = now_ms() - start_ms;
    } while (elapsed < kMinTimeMs);

    double mops = (static_cast<double>(N) * static_cast<double>(iters))
                  / (elapsed * 1e3);  /* elapsed in ms → × 1e3 gives us Mops/s */

    g_results.push_back({name, mops, elapsed, iters});
}

/* ── Aligned buffer helper ───────────────────────────────────────────────── */

/*
 * Simple 64-byte aligned buffer allocation backed by a vector<float>.
 * The pointer is stable for the lifetime of the object.
 */
struct AlignedBuf {
    std::vector<float> storage;
    float*             ptr;

    explicit AlignedBuf(size_t n, float fill = 0.0f) {
        /* Over-allocate by 16 floats to guarantee 64-byte alignment. */
        storage.resize(n + 16, fill);
        uintptr_t raw  = reinterpret_cast<uintptr_t>(storage.data());
        uintptr_t aligned = (raw + 63u) & ~static_cast<uintptr_t>(63u);
        ptr = reinterpret_cast<float*>(aligned);
    }
};

/* =========================================================================
 * SVE Predicate Construction Benchmarks
 * ========================================================================= */

static void bench_predicates() {
    /* pred_all_f32 — trivial but measures call overhead */
    run_bench("SvePred/pred_all_f32", 1, [] {
        volatile auto pg = navexa::sve::pred_all_f32();
        (void)pg;
    });

    run_bench("SvePred/pred_first_n", 1, [] {
        volatile auto pg = navexa::sve::pred_first_n(3);
        (void)pg;
    });

    run_bench("SvePred/pred_while_lt", 1, [] {
        volatile auto pg = navexa::sve::pred_while_lt(4, 7);
        (void)pg;
    });
}

/* =========================================================================
 * SVE Predicated Load / Store Benchmarks
 * ========================================================================= */

static void bench_load_store() {
    constexpr size_t N = kMedN;
    AlignedBuf src(N, 1.0f);
    AlignedBuf dst(N, 0.0f);

    /* Full-predicate load (all 4 lanes) */
    run_bench("SveLoad/load_f32_full_N" + std::to_string(N), N, [&] {
        navexa::sve::SvePred pg = navexa::sve::pred_all_f32();
        for (size_t i = 0; i < N; i += 4) {
            float32x4_t v = navexa::sve::load_f32(src.ptr + i, pg);
            do_not_optimize_f32(v);
        }
    });

    /* Partial-predicate load (3 of 4 lanes) — exercises blending path */
    run_bench("SveLoad/load_f32_partial_N" + std::to_string(N), N, [&] {
        navexa::sve::SvePred pg = navexa::sve::pred_first_n(3);
        for (size_t i = 0; i < N; i += 4) {
            float32x4_t v = navexa::sve::load_f32(src.ptr + i, pg);
            do_not_optimize_f32(v);
        }
    });

    /* Store all lanes */
    run_bench("SveStore/store_f32_full_N" + std::to_string(N), N, [&] {
        navexa::sve::SvePred pg = navexa::sve::pred_all_f32();
        float32x4_t v = vdupq_n_f32(1.0f);
        for (size_t i = 0; i < N; i += 4) {
            navexa::sve::store_f32(dst.ptr + i, v, pg);
        }
        do_not_optimize(dst.ptr, N);
    });

    /* Store partial lanes */
    run_bench("SveStore/store_f32_partial_N" + std::to_string(N), N, [&] {
        navexa::sve::SvePred pg = navexa::sve::pred_first_n(2);
        float32x4_t v = vdupq_n_f32(2.0f);
        for (size_t i = 0; i < N; i += 4) {
            navexa::sve::store_f32(dst.ptr + i, v, pg);
        }
        do_not_optimize(dst.ptr, N);
    });
}

/* =========================================================================
 * SVE Predicated Arithmetic Benchmarks
 * ========================================================================= */

static void bench_sve_arith() {
    constexpr size_t N = kMedN;
    AlignedBuf a(N, 1.5f);
    AlignedBuf b(N, 0.5f);
    float32x4_t va = vld1q_f32(a.ptr);
    float32x4_t vb = vld1q_f32(b.ptr);
    float dst[4];

    /* All-lane add */
    run_bench("SveArith/add_f32_full", 1, [&] {
        float32x4_t r = navexa::sve::add_f32(va, vb, navexa::sve::pred_all_f32());
        do_not_optimize_f32(r);
    });

    /* Partial-lane add (merging path) */
    navexa::sve::SvePred pg;
    pg.mask = 0b0101;
    run_bench("SveArith/add_f32_partial", 1, [&] {
        float32x4_t r = navexa::sve::add_f32(va, vb, pg);
        do_not_optimize_f32(r);
    });

    run_bench("SveArith/sub_f32_full", 1, [&] {
        float32x4_t r = navexa::sve::sub_f32(va, vb, navexa::sve::pred_all_f32());
        do_not_optimize_f32(r);
    });

    run_bench("SveArith/mul_f32_full", 1, [&] {
        float32x4_t r = navexa::sve::mul_f32(va, vb, navexa::sve::pred_all_f32());
        do_not_optimize_f32(r);
    });

    /* FMA: a*b + c */
    float32x4_t vc = vdupq_n_f32(0.1f);
    run_bench("SveArith/fma_f32_full", 1, [&] {
        float32x4_t r = navexa::sve::fma_f32(va, vb, vc, navexa::sve::pred_all_f32());
        do_not_optimize_f32(r);
    });

    /* f64 operations */
    double ad[2] = {1.5, 2.5}, bd[2] = {3.5, 4.5};
    float64x2_t vad = vld1q_f64(ad), vbd = vld1q_f64(bd);

    run_bench("SveArith/add_f64_full", 1, [&] {
        float64x2_t r = navexa::sve::add_f64(vad, vbd, navexa::sve::pred_all_f64());
        volatile double buf[2];
        vst1q_f64(const_cast<double*>(buf), r);
    });

    run_bench("SveArith/mul_f64_full", 1, [&] {
        float64x2_t r = navexa::sve::mul_f64(vad, vbd, navexa::sve::pred_all_f64());
        volatile double buf[2];
        vst1q_f64(const_cast<double*>(buf), r);
    });
}

/* =========================================================================
 * SVE Horizontal Reduction Benchmarks
 * ========================================================================= */

static void bench_sve_reduce() {
    float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float32x4_t v = vld1q_f32(vals);

    run_bench("SveReduce/reduce_add_f32_full", 1, [&] {
        volatile float r = navexa::sve::reduce_add_f32(v, navexa::sve::pred_all_f32());
        (void)r;
    });

    run_bench("SveReduce/reduce_max_f32_full", 1, [&] {
        volatile float r = navexa::sve::reduce_max_f32(v, navexa::sve::pred_all_f32());
        (void)r;
    });

    navexa::sve::SvePred pg = navexa::sve::pred_first_n(2);

    run_bench("SveReduce/reduce_add_f32_partial", 1, [&] {
        volatile float r = navexa::sve::reduce_add_f32(v, pg);
        (void)r;
    });

    run_bench("SveReduce/reduce_max_f32_partial", 1, [&] {
        volatile float r = navexa::sve::reduce_max_f32(v, pg);
        (void)r;
    });
}

/* =========================================================================
 * SVE Gather / Scatter Benchmarks
 * ========================================================================= */

static void bench_sve_gather_scatter() {
    constexpr size_t BASE_SIZE = 512;
    std::vector<float> base(BASE_SIZE);
    for (size_t i = 0; i < BASE_SIZE; ++i) base[i] = static_cast<float>(i);

    /* Stride-4 gather pattern: base[0,4,8,12] */
    int32_t stride4_idx[4] = {0, 4, 8, 12};

    /* Random-ish (but cache-warm) gather: spread across the array */
    int32_t spread_idx[4] = {0, 128, 256, 384};

    run_bench("SveGather/gather_stride4", 4, [&] {
        navexa::sve::SvePred pg = navexa::sve::pred_all_f32();
        float32x4_t r = navexa::sve::gather_f32(base.data(), stride4_idx, pg);
        do_not_optimize_f32(r);
    });

    run_bench("SveGather/gather_spread", 4, [&] {
        navexa::sve::SvePred pg = navexa::sve::pred_all_f32();
        float32x4_t r = navexa::sve::gather_f32(base.data(), spread_idx, pg);
        do_not_optimize_f32(r);
    });

    run_bench("SveGather/gather_partial_2lanes", 4, [&] {
        navexa::sve::SvePred pg = navexa::sve::pred_first_n(2);
        float32x4_t r = navexa::sve::gather_f32(base.data(), stride4_idx, pg);
        do_not_optimize_f32(r);
    });

    /* Scatter benchmarks */
    std::vector<float> dest(BASE_SIZE, 0.0f);
    float32x4_t scatter_vals = vdupq_n_f32(99.0f);

    run_bench("SveScatter/scatter_stride4", 4, [&] {
        navexa::sve::SvePred pg = navexa::sve::pred_all_f32();
        navexa::sve::scatter_f32(dest.data(), stride4_idx, scatter_vals, pg);
    });

    run_bench("SveScatter/scatter_spread", 4, [&] {
        navexa::sve::SvePred pg = navexa::sve::pred_all_f32();
        navexa::sve::scatter_f32(dest.data(), spread_idx, scatter_vals, pg);
    });
}

/* =========================================================================
 * SVE vecop_f32 Loop Benchmarks
 * ========================================================================= */

static void bench_sve_vecop() {
    for (size_t N : {kMedN, kLargeN}) {
        AlignedBuf a(N, 1.0f);
        AlignedBuf b(N, 0.5f);
        AlignedBuf out(N, 0.0f);
        std::string suffix = "_N" + std::to_string(N);

        run_bench("SveVecop/ADD" + suffix, N, [&] {
            navexa::sve::vecop_f32(out.ptr, a.ptr, b.ptr, N,
                                   navexa::sve::SveOp::ADD);
            do_not_optimize(out.ptr, N);
        });

        run_bench("SveVecop/SUB" + suffix, N, [&] {
            navexa::sve::vecop_f32(out.ptr, a.ptr, b.ptr, N,
                                   navexa::sve::SveOp::SUB);
            do_not_optimize(out.ptr, N);
        });

        run_bench("SveVecop/MUL" + suffix, N, [&] {
            navexa::sve::vecop_f32(out.ptr, a.ptr, b.ptr, N,
                                   navexa::sve::SveOp::MUL);
            do_not_optimize(out.ptr, N);
        });

        /* Non-multiple-of-4 tail exercising pred_while_lt */
        const size_t N_odd = N - 3;
        run_bench("SveVecop/ADD_odd" + suffix, N_odd, [&] {
            navexa::sve::vecop_f32(out.ptr, a.ptr, b.ptr, N_odd,
                                   navexa::sve::SveOp::ADD);
            do_not_optimize(out.ptr, N_odd);
        });
    }
}

/* =========================================================================
 * MathExt: Exponential and Logarithm Benchmarks
 * ========================================================================= */

static void bench_math_exp_log() {
    for (size_t N : {kMedN, kLargeN}) {
        AlignedBuf in(N);
        AlignedBuf out(N);
        std::string suffix = "_N" + std::to_string(N);

        /* Fill with values in [-5, 5] to keep exp in a reasonable range */
        for (size_t i = 0; i < N; ++i)
            in.ptr[i] = -5.0f + 10.0f * (static_cast<float>(i) / static_cast<float>(N));

        run_bench("MathExp/vec_exp_f32" + suffix, N, [&] {
            navexa::math::vec_exp_f32(out.ptr, in.ptr, N);
            do_not_optimize(out.ptr, N);
        });

        /* For log: fill with positive values */
        for (size_t i = 0; i < N; ++i)
            in.ptr[i] = 0.1f + static_cast<float>(i);

        run_bench("MathExp/vec_log_f32" + suffix, N, [&] {
            navexa::math::vec_log_f32(out.ptr, in.ptr, N);
            do_not_optimize(out.ptr, N);
        });

        /* Reset in for exp2 */
        for (size_t i = 0; i < N; ++i)
            in.ptr[i] = -5.0f + 10.0f * (static_cast<float>(i) / static_cast<float>(N));

        run_bench("MathExp/vec_exp2_f32" + suffix, N, [&] {
            navexa::math::vec_exp2_f32(out.ptr, in.ptr, N);
            do_not_optimize(out.ptr, N);
        });
    }
}

/* =========================================================================
 * MathExt: Trigonometry Benchmarks
 * ========================================================================= */

static void bench_math_trig() {
    for (size_t N : {kMedN, kLargeN}) {
        AlignedBuf in(N);
        AlignedBuf s(N), c(N);
        std::string suffix = "_N" + std::to_string(N);

        /* Fill with angles in [0, 2π] */
        for (size_t i = 0; i < N; ++i)
            in.ptr[i] = 6.28318f * (static_cast<float>(i) / static_cast<float>(N));

        run_bench("MathTrig/vec_sin_f32" + suffix, N, [&] {
            navexa::math::vec_sin_f32(s.ptr, in.ptr, N);
            do_not_optimize(s.ptr, N);
        });

        run_bench("MathTrig/vec_cos_f32" + suffix, N, [&] {
            navexa::math::vec_cos_f32(c.ptr, in.ptr, N);
            do_not_optimize(c.ptr, N);
        });

        /* vec_sincos is faster than sin + cos separately */
        run_bench("MathTrig/vec_sincos_f32" + suffix, N, [&] {
            navexa::math::vec_sincos_f32(s.ptr, c.ptr, in.ptr, N);
            do_not_optimize(s.ptr, N);
            do_not_optimize(c.ptr, N);
        });
    }
}

/* =========================================================================
 * MathExt: Activation Functions Benchmarks
 * ========================================================================= */

static void bench_math_activations() {
    for (size_t N : {kMedN, kLargeN}) {
        AlignedBuf in(N);
        AlignedBuf out(N);
        std::string suffix = "_N" + std::to_string(N);

        /* Fill with values typical for ML activations */
        for (size_t i = 0; i < N; ++i)
            in.ptr[i] = -4.0f + 8.0f * (static_cast<float>(i) / static_cast<float>(N));

        run_bench("MathActiv/vec_tanh_f32" + suffix, N, [&] {
            navexa::math::vec_tanh_f32(out.ptr, in.ptr, N);
            do_not_optimize(out.ptr, N);
        });

        run_bench("MathActiv/vec_sigmoid_f32" + suffix, N, [&] {
            navexa::math::vec_sigmoid_f32(out.ptr, in.ptr, N);
            do_not_optimize(out.ptr, N);
        });

        run_bench("MathActiv/vec_relu_f32" + suffix, N, [&] {
            navexa::math::vec_relu_f32(out.ptr, in.ptr, N);
            do_not_optimize(out.ptr, N);
        });
    }
}

/* =========================================================================
 * MathExt: General Vector Operations Benchmarks
 * ========================================================================= */

static void bench_math_vecops() {
    for (size_t N : {kMedN, kLargeN}) {
        AlignedBuf a(N, 1.5f);
        AlignedBuf b(N, 0.3f);
        AlignedBuf c(N, 0.1f);
        AlignedBuf out(N);
        std::string suffix = "_N" + std::to_string(N);

        run_bench("MathVecOp/vec_add_f32" + suffix, N, [&] {
            navexa::math::vec_add_f32(out.ptr, a.ptr, b.ptr, N);
            do_not_optimize(out.ptr, N);
        });

        run_bench("MathVecOp/vec_mul_f32" + suffix, N, [&] {
            navexa::math::vec_mul_f32(out.ptr, a.ptr, b.ptr, N);
            do_not_optimize(out.ptr, N);
        });

        run_bench("MathVecOp/vec_fma_f32" + suffix, N, [&] {
            navexa::math::vec_fma_f32(out.ptr, a.ptr, b.ptr, c.ptr, N);
            do_not_optimize(out.ptr, N);
        });

        run_bench("MathVecOp/vec_dot_f32" + suffix, N, [&] {
            volatile float dot = navexa::math::vec_dot_f32(a.ptr, b.ptr, N);
            (void)dot;
        });

        /* Fill with positive values for sqrt */
        for (size_t i = 0; i < N; ++i) a.ptr[i] = static_cast<float>(i + 1);

        run_bench("MathVecOp/vec_sqrt_f32" + suffix, N, [&] {
            navexa::math::vec_sqrt_f32(out.ptr, a.ptr, N);
            do_not_optimize(out.ptr, N);
        });
    }
}

/* =========================================================================
 * MathExt: Fixed-Point Conversion Benchmarks
 * ========================================================================= */

static void bench_math_fixedpoint() {
    for (size_t N : {kMedN, kLargeN}) {
        AlignedBuf in_f(N);
        std::vector<int16_t> q15(N + 16);
        /* Align q15 storage too */
        int16_t* q15_ptr = reinterpret_cast<int16_t*>(
            (reinterpret_cast<uintptr_t>(q15.data()) + 31u) & ~static_cast<uintptr_t>(31u));
        AlignedBuf out_f(N);
        std::string suffix = "_N" + std::to_string(N);

        /* Fill with values in [-1, 1] (Q1.15 range for scale=1) */
        for (size_t i = 0; i < N; ++i)
            in_f.ptr[i] = -1.0f + 2.0f * (static_cast<float>(i) / static_cast<float>(N));

        run_bench("MathFixed/f32_to_q15" + suffix, N, [&] {
            navexa::math::f32_to_q15(q15_ptr, in_f.ptr, N, 1.0f);
            volatile int16_t x = q15_ptr[0];
            (void)x;
        });

        /* Pre-fill q15 for the reverse benchmark */
        navexa::math::f32_to_q15(q15_ptr, in_f.ptr, N, 1.0f);

        run_bench("MathFixed/q15_to_f32" + suffix, N, [&] {
            navexa::math::q15_to_f32(out_f.ptr, q15_ptr, N, 1.0f);
            do_not_optimize(out_f.ptr, N);
        });

        /* Round-trip: f32→q15→f32 (full pipeline) */
        run_bench("MathFixed/f32_q15_roundtrip" + suffix, N, [&] {
            navexa::math::f32_to_q15(q15_ptr, in_f.ptr, N, 1.0f);
            navexa::math::q15_to_f32(out_f.ptr, q15_ptr, N, 1.0f);
            do_not_optimize(out_f.ptr, N);
        });
    }
}

/* =========================================================================
 * SVE Emulation Detection Benchmark
 * ========================================================================= */

static void bench_sve_detection() {
    run_bench("SveDetect/is_hw_available", 1, [] {
        volatile bool r = navexa::sve::is_hw_available();
        (void)r;
    });

    run_bench("SveDetect/get_emulated_vl", 1, [] {
        volatile size_t r = navexa::sve::get_emulated_vl();
        (void)r;
    });
}

/* =========================================================================
 * Composite Throughput Benchmarks
 *
 * These replicate real workloads that cross the boundary between SVE
 * predication and mathext to expose any inter-module overhead.
 * ========================================================================= */

static void bench_composite() {
    constexpr size_t N = kMedN;
    AlignedBuf a(N), b(N), out(N);

    for (size_t i = 0; i < N; ++i) {
        a.ptr[i] =  0.5f + 0.01f * static_cast<float>(i % 16);
        b.ptr[i] = -0.5f + 0.01f * static_cast<float>(i % 8);
    }

    /* Benchmark 1: SVE ADD → exp (simulates pre-activation shifting) */
    AlignedBuf sve_sum(N, 0.0f);
    run_bench("Composite/SveAdd_then_Exp_N" + std::to_string(N), N, [&] {
        navexa::sve::vecop_f32(sve_sum.ptr, a.ptr, b.ptr, N,
                               navexa::sve::SveOp::ADD);
        navexa::math::vec_exp_f32(out.ptr, sve_sum.ptr, N);
        do_not_optimize(out.ptr, N);
    });

    /* Benchmark 2: SVE MUL → tanh (simulates attention score + activation) */
    run_bench("Composite/SveMul_then_Tanh_N" + std::to_string(N), N, [&] {
        navexa::sve::vecop_f32(sve_sum.ptr, a.ptr, b.ptr, N,
                               navexa::sve::SveOp::MUL);
        navexa::math::vec_tanh_f32(out.ptr, sve_sum.ptr, N);
        do_not_optimize(out.ptr, N);
    });

    /* Benchmark 3: sin + cos + dot product (physics / rotation workload) */
    AlignedBuf s(N), c(N), ones(N, 1.0f);
    run_bench("Composite/SinCos_then_Dot_N" + std::to_string(N), N, [&] {
        navexa::math::vec_sincos_f32(s.ptr, c.ptr, a.ptr, N);
        volatile float d = navexa::math::vec_dot_f32(s.ptr, c.ptr, N);
        (void)d;
    });

    /* Benchmark 4: exp → normalize (softmax inner loop) */
    AlignedBuf exp_out(N, 0.0f);
    run_bench("Composite/Exp_Normalize_N" + std::to_string(N), N, [&] {
        navexa::math::vec_exp_f32(exp_out.ptr, a.ptr, N);
        float ones_arr[1] = {1.0f};
        float sum_arr[1] = {};
        navexa::math::vec_dot_f32(exp_out.ptr, ones.ptr, N);  /* sum of exp */
        do_not_optimize(exp_out.ptr, N);
    });
}

/* =========================================================================
 * Report
 * ========================================================================= */

static void print_header() {
    printf("\n");
    printf("==================================================================="
           "=============\n");
    printf("  ARMv8-A Library — SVE + MathExt Micro-Benchmarks\n");
    printf("  Arch: %s | SVE HW: %s | Emulated VL: %zu bits\n",
#if defined(__aarch64__)
           "aarch64",
#else
           "non-aarch64 (results may be emulated)",
#endif
           navexa::sve::is_hw_available() ? "yes" : "no (NEON emulation)",
           navexa::sve::get_emulated_vl());
    printf("  Timing: min %.0f ms per benchmark | warm-up: %d iter\n",
           kMinTimeMs, kWarmupIter);
    printf("==================================================================="
           "=============\n");
    printf("  %-55s  %12s  %8s  %8s\n",
           "Benchmark", "MOps/s", "ms", "iters");
    printf("  %-55s  %12s  %8s  %8s\n",
           std::string(55, '-').c_str(), "----------",
           "------", "------");
}

static void print_results() {
    /* Group by prefix (text before first '/') */
    std::string last_prefix;
    for (const auto& r : g_results) {
        std::string prefix = r.name.substr(0, r.name.find('/'));
        if (prefix != last_prefix) {
            printf("\n");
            last_prefix = prefix;
        }
        printf("  %-55s  %12.2f  %8.1f  %8lld\n",
               r.name.c_str(), r.mops, r.ms_total, r.iters);
    }
    printf("\n");

    /* Summary: fastest and slowest */
    if (g_results.size() >= 2) {
        auto [lo, hi] = std::minmax_element(
            g_results.begin(), g_results.end(),
            [](const BenchResult& a, const BenchResult& b) {
                return a.mops < b.mops;
            });
        printf("  Fastest: %-48s %.2f MOps/s\n", hi->name.c_str(), hi->mops);
        printf("  Slowest: %-48s %.2f MOps/s\n", lo->name.c_str(), lo->mops);
        printf("\n");
    }
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(int argc, char* argv[]) {
    if (argc > 1) {
        g_filter = argv[1];
        printf("Filter: \"%s\"\n", g_filter.c_str());
    }

    print_header();

    /* Run all benchmark groups */
    bench_sve_detection();
    bench_predicates();
    bench_load_store();
    bench_sve_arith();
    bench_sve_reduce();
    bench_sve_gather_scatter();
    bench_sve_vecop();
    bench_math_exp_log();
    bench_math_trig();
    bench_math_activations();
    bench_math_vecops();
    bench_math_fixedpoint();
    bench_composite();

    print_results();

    printf("  Total benchmarks run: %zu\n\n", g_results.size());
    return 0;
}

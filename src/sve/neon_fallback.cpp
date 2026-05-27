/*
 * src/sve/neon_fallback.cpp
 * Pure NEON implementations — no SVE predication overhead
 *
 * These are the "inner loop" kernels that sve_emulation.cpp delegates to
 * when the full array length is a multiple of 4 (the common case in HPC/ML).
 * They use raw NEON without any scalar blending, making them faster for
 * bulk operations on aligned, full-width data.
 *
 * Not part of the public API directly — called internally by sve_emulation.cpp
 * and by mathext/transcendentals.cpp for its vectorized inner loops.
 *
 * Exposed via an internal header: src/sve/neon_fallback.h (not in include/)
 */

#include "neon_fallback.h"
#include <arm_neon.h>
#include <cstring>  /* memcpy */

namespace navexa {
namespace neon {

/* -------------------------------------------------------------------------
 * Arithmetic bulk operations (len must be multiple of 4)
 * ------------------------------------------------------------------------- */

void add_f32_bulk(float* out, const float* a, const float* b, size_t len) {
    /*
     * Process 4 elements per iteration.
     * vld1q_f32: load 4 floats into a 128-bit NEON register (Q-register)
     * vaddq_f32: add two Q-registers lane-wise, single cycle throughput
     * vst1q_f32: store Q-register back to memory
     *
     * len assumed multiple of 4 — caller handles tail via SVE predication.
     */
    for (size_t i = 0; i < len; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
}

void sub_f32_bulk(float* out, const float* a, const float* b, size_t len) {
    for (size_t i = 0; i < len; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vsubq_f32(va, vb));
    }
}

void mul_f32_bulk(float* out, const float* a, const float* b, size_t len) {
    for (size_t i = 0; i < len; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
}

void fma_f32_bulk(float* out, const float* a, const float* b,
                  const float* c, size_t len) {
    /*
     * vfmaq_f32(c, a, b) = a*b + c
     * Single fused instruction: no intermediate rounding between mul and add.
     * ARM guarantees this is truly fused (not mul+add) on all ARMv8-A cores.
     */
    for (size_t i = 0; i < len; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        /* vfmaq_f32(addend, mul1, mul2): result = mul1*mul2 + addend */
        vst1q_f32(out + i, vfmaq_f32(vc, va, vb));
    }
}

/* -------------------------------------------------------------------------
 * Dot product
 * ------------------------------------------------------------------------- */

float dot_f32_bulk(const float* a, const float* b, size_t len) {
    /*
     * Accumulate partial sums in a NEON vector to maximise throughput.
     * At the end, reduce the 4-lane accumulator to a scalar.
     *
     * This is pairwise reduction:
     *   acc = [s0, s1, s2, s3]
     *   vpadd_f32([s0,s1], [s2,s3]) = [s0+s1, s2+s3]
     *   then add those two scalars
     */
    float32x4_t vacc = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < len; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        /* vfmaq_f32: vacc += va * vb (fused, one instruction) */
        vacc = vfmaq_f32(vacc, va, vb);
    }

    /* Reduce 4-lane accumulator to scalar */
    float32x2_t lo  = vget_low_f32(vacc);
    float32x2_t hi  = vget_high_f32(vacc);
    float32x2_t sum = vpadd_f32(lo, hi);     /* [acc0+acc1, acc2+acc3] */
    return vget_lane_f32(sum, 0) + vget_lane_f32(sum, 1);
}

/* -------------------------------------------------------------------------
 * Square root via Newton-Raphson refinement
 * ------------------------------------------------------------------------- */

void sqrt_f32_bulk(float* out, const float* in, size_t len) {
    /*
     * ARM approach: compute 1/sqrt(x) first, then multiply by x.
     *
     * vrsqrteq_f32: reciprocal square root ESTIMATE — fast but ~8-bit accurate.
     * vrsqrtsq_f32: Newton-Raphson step for refining rsqrt estimate.
     *   vrsqrtsq_f32(a, b) = (3 - a*b) / 2   (the NR correction formula)
     *
     * Two NR steps give us full f32 precision (~23 bits).
     *
     * Step 1: e0 = rsqrt_estimate(x)          ~8-bit accurate
     * Step 2: e1 = e0 * NR_step(x, e0*e0)     ~16-bit accurate
     * Step 3: e2 = e1 * NR_step(x, e1*e1)     ~23-bit accurate (full f32)
     * Final:  sqrt(x) = x * e2                (x * (1/sqrt(x)) = sqrt(x))
     */
    for (size_t i = 0; i < len; i += 4) {
        float32x4_t vx = vld1q_f32(in + i);

        /* Initial estimate of 1/sqrt(x) */
        float32x4_t e = vrsqrteq_f32(vx);

        /* First Newton-Raphson refinement step */
        /* vrsqrtsq_f32(vx, e*e) computes (3 - vx*e*e) / 2 */
        float32x4_t e_sq = vmulq_f32(e, e);
        e = vmulq_f32(e, vrsqrtsq_f32(vx, e_sq));

        /* Second Newton-Raphson refinement step */
        e_sq = vmulq_f32(e, e);
        e = vmulq_f32(e, vrsqrtsq_f32(vx, e_sq));

        /* sqrt(x) = x * (1/sqrt(x)) */
        vst1q_f32(out + i, vmulq_f32(vx, e));
    }
}

/* -------------------------------------------------------------------------
 * ReLU
 * ------------------------------------------------------------------------- */

void relu_f32_bulk(float* out, const float* in, size_t len) {
    /*
     * vmaxq_f32(v, zero): per-lane max with 0.
     * Single instruction, no branches, handles negative/positive/NaN correctly.
     */
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < len; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, vmaxq_f32(v, zero));
    }
}

/* -------------------------------------------------------------------------
 * Fixed-point conversion
 * ------------------------------------------------------------------------- */

void f32_to_q15_bulk(int16_t* out, const float* in, size_t len, float scale) {
    /*
     * Q1.15: multiply by (scale * 32768), round, clamp to [-32768, 32767].
     *
     * vcvtq_n_s32_f32(v, 15): convert float to int32 with 15 fractional bits
     * vmovn_s32: narrow int32x4 → int16x4 (truncates to 16 bits)
     *
     * We process 4 f32 → 4 int16 per iteration.
     */
    float32x4_t vscale = vdupq_n_f32(scale * 32768.0f);

    for (size_t i = 0; i < len; i += 4) {
        float32x4_t vf = vld1q_f32(in + i);

        /* Scale */
        float32x4_t vscaled = vmulq_f32(vf, vscale);

        /* Convert to int32 with rounding (vcvtnq = round-to-nearest) */
        int32x4_t vi32 = vcvtnq_s32_f32(vscaled);

        /* Clamp to int16 range using saturating narrow */
        int16x4_t vi16 = vqmovn_s32(vi32); /* saturating narrow: clamps to [-32768,32767] */

        vst1_s16(out + i, vi16);
    }
}

void q15_to_f32_bulk(float* out, const int16_t* in, size_t len, float scale) {
    /*
     * Convert int16 Q1.15 back to float.
     * vmovl_s16: widen int16x4 → int32x4 (sign-extend)
     * vcvtq_f32_s32: convert int32 to float
     * Then divide by (scale * 32768).
     */
    float32x4_t inv_scale = vdupq_n_f32(1.0f / (scale * 32768.0f));

    for (size_t i = 0; i < len; i += 4) {
        int16x4_t  vi16 = vld1_s16(in + i);
        int32x4_t  vi32 = vmovl_s16(vi16);         /* sign-extend to 32-bit */
        float32x4_t vf  = vcvtq_f32_s32(vi32);     /* int32 → float */
        vst1q_f32(out + i, vmulq_f32(vf, inv_scale));
    }
}

} // namespace neon
} // namespace navexa
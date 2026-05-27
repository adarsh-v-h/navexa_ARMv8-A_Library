/*
 * src/sve/sve_emulation.cpp
 * SVE emulation layer — predication, gather/scatter, vecop loop helper
 *
 * All SVE semantics are emulated at 128-bit width using NEON.
 * See include/armv8lib/sve.h for the full API contract.
 */

#include "armv8lib/sve.h"

#include <arm_neon.h>
#include <sys/auxv.h>   /* getauxval */
#include <cstring>      /* memcpy */
#include <cmath>        /* INFINITY */
#ifdef __aarch64__
    #include <sys/auxv.h>
    #include <asm/hwcap.h>
#endif


namespace navexa {
namespace sve {

/* -------------------------------------------------------------------------
 * Detection
 * ------------------------------------------------------------------------- */

bool is_hw_available() {
#if defined(__aarch64__) && defined(HWCAP2_SVE)
    return (getauxval(AT_HWCAP2) & HWCAP2_SVE) != 0;
#else
    return false;
#endif
}

size_t get_emulated_vl() {
    /* Always 128-bit: 4 x f32 or 2 x f64 per vector register */
    return 128;
}

/* -------------------------------------------------------------------------
 * Predicate construction
 * ------------------------------------------------------------------------- */

SvePred pred_all_f32() {
    /* All 4 lanes active: bits 0-3 set = 0b00001111 = 0x0F */
    return SvePred{0x0F};
}

SvePred pred_all_f64() {
    /* All 2 lanes active: bits 0-1 set = 0b00000011 = 0x03 */
    return SvePred{0x03};
}

SvePred pred_first_n(size_t n) {
    /*
     * Build mask with the bottom n bits set.
     * n=0 → 0x00, n=1 → 0x01, n=2 → 0x03, n=3 → 0x07, n=4 → 0x0F
     * Clamp n to [0, 4] for f32.
     *
     * (1u << n) - 1 gives us a mask of n set bits.
     * Special case n=0 to avoid UB (1u << 0) - 1 = 0 is fine,
     * but n >= 8 would overflow uint8_t — clamp first.
     */
    if (n == 0) return SvePred{0x00};
    if (n >= 4) return SvePred{0x0F};
    return SvePred{static_cast<uint8_t>((1u << n) - 1u)};
}

SvePred pred_while_lt(size_t idx, size_t limit) {
    /*
     * Lane k is active if (idx + k) < limit.
     * Equivalent to SVE whilelt(idx, limit).
     *
     * Example: idx=6, limit=7, 4-wide vector
     *   lane 0: 6 < 7 → active
     *   lane 1: 7 < 7 → inactive
     *   lane 2: 8 < 7 → inactive
     *   lane 3: 9 < 7 → inactive
     *   mask = 0b0001 = 0x01
     */
    uint8_t mask = 0;
    for (int k = 0; k < 4; ++k) {
        if ((idx + static_cast<size_t>(k)) < limit) {
            mask |= static_cast<uint8_t>(1u << k);
        }
    }
    return SvePred{mask};
}

/* -------------------------------------------------------------------------
 * Predicated load / store
 * ------------------------------------------------------------------------- */

SveVecF32 load_f32(const float* ptr, SvePred pg) {
    /*
     * We cannot use vld1q_f32 unconditionally if the predicate excludes
     * some lanes — the pointer may not have data there.
     * Safe approach: load into a local buffer lane-by-lane, then build
     * the NEON register. For the common all-active case this is a fast path.
     */
    float buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        if (pg.active(i)) {
            buf[i] = ptr[i];
        }
    }
    return vld1q_f32(buf);
}

void store_f32(float* ptr, SveVecF32 v, SvePred pg) {
    /*
     * Store only active lanes. Inactive lanes must NOT write to memory.
     * Extract all 4 lanes, write only the ones the predicate says to.
     */
    float buf[4];
    vst1q_f32(buf, v);
    for (int i = 0; i < 4; ++i) {
        if (pg.active(i)) {
            ptr[i] = buf[i];
        }
    }
}

SveVecF64 load_f64(const double* ptr, SvePred pg) {
    double buf[2] = {0.0, 0.0};
    for (int i = 0; i < 2; ++i) {
        if (pg.active(i)) {
            buf[i] = ptr[i];
        }
    }
    return vld1q_f64(buf);
}

void store_f64(double* ptr, SveVecF64 v, SvePred pg) {
    double buf[2];
    vst1q_f64(buf, v);
    for (int i = 0; i < 2; ++i) {
        if (pg.active(i)) {
            ptr[i] = buf[i];
        }
    }
}

/* -------------------------------------------------------------------------
 * Predicated arithmetic (merging)
 *
 * Merging predication: active lanes → compute. Inactive lanes → keep 'a'.
 * We build a NEON bitmask from SvePred and use vbslq_f32 (bitselect) to blend.
 * vbslq_f32(mask, true_val, false_val): mask bit=1 → true_val, bit=0 → false_val
 * ------------------------------------------------------------------------- */

/*
 * Internal helper: expand SvePred into a NEON lane mask for vbslq_f32.
 * Each active lane becomes 0xFFFFFFFF, each inactive lane becomes 0x00000000.
 */
static uint32x4_t make_neon_mask(SvePred pg) {
    /*
     * Build each 32-bit lane of the mask from the corresponding bit in pg.mask.
     * Active (bit=1) → all 32 bits set (0xFFFFFFFF = UINT32_MAX)
     * Inactive (bit=0) → all 32 bits clear (0x00000000)
     */
    uint32_t m[4];
    for (int i = 0; i < 4; ++i) {
        m[i] = pg.active(i) ? 0xFFFFFFFFu : 0x00000000u;
    }
    return vld1q_u32(m);
}

SveVecF32 add_f32(SveVecF32 a, SveVecF32 b, SvePred pg) {
    SveVecF32 result = vaddq_f32(a, b);
    /* Blend: active lanes take result, inactive lanes keep a */
    uint32x4_t mask = make_neon_mask(pg);
    return vbslq_f32(mask, result, a);
}

SveVecF32 sub_f32(SveVecF32 a, SveVecF32 b, SvePred pg) {
    SveVecF32 result = vsubq_f32(a, b);
    uint32x4_t mask = make_neon_mask(pg);
    return vbslq_f32(mask, result, a);
}

SveVecF32 mul_f32(SveVecF32 a, SveVecF32 b, SvePred pg) {
    SveVecF32 result = vmulq_f32(a, b);
    uint32x4_t mask = make_neon_mask(pg);
    return vbslq_f32(mask, result, a);
}

SveVecF32 fma_f32(SveVecF32 a, SveVecF32 b, SveVecF32 c, SvePred pg) {
    /*
     * vfmaq_f32(c, a, b) = a*b + c  (fused, single instruction on ARMv8-A)
     * Note: ARM's vfmaq_f32 argument order is (addend, mul1, mul2)
     */
    SveVecF32 result = vfmaq_f32(c, a, b);
    uint32x4_t mask = make_neon_mask(pg);
    return vbslq_f32(mask, result, a);
}

SveVecF64 add_f64(SveVecF64 a, SveVecF64 b, SvePred pg) {
    SveVecF64 result = vaddq_f64(a, b);
    /* f64: 2 lanes, use bits 0-1 of pg.mask */
    uint64_t m[2];
    m[0] = pg.active(0) ? UINT64_MAX : 0ULL;
    m[1] = pg.active(1) ? UINT64_MAX : 0ULL;
    uint64x2_t mask = vld1q_u64(m);
    return vbslq_f64(mask, result, a);
}

SveVecF64 mul_f64(SveVecF64 a, SveVecF64 b, SvePred pg) {
    SveVecF64 result = vmulq_f64(a, b);
    uint64_t m[2];
    m[0] = pg.active(0) ? UINT64_MAX : 0ULL;
    m[1] = pg.active(1) ? UINT64_MAX : 0ULL;
    uint64x2_t mask = vld1q_u64(m);
    return vbslq_f64(mask, result, a);
}

/* -------------------------------------------------------------------------
 * Horizontal reduction
 * ------------------------------------------------------------------------- */

float reduce_add_f32(SveVecF32 v, SvePred pg) {
    /*
     * Zero out inactive lanes before summing.
     * vbslq_f32: active lanes keep v, inactive lanes become 0.0f
     */
    uint32x4_t mask = make_neon_mask(pg);
    SveVecF32 zeroed = vbslq_f32(mask, v, vdupq_n_f32(0.0f));

    /*
     * NEON pairwise horizontal add:
     * vpadd_f32 operates on 64-bit halves (2 lanes each).
     * Step 1: pair up [0+1, 2+3]  (using the low and high halves)
     * Step 2: add those two results
     */
    float32x2_t lo = vget_low_f32(zeroed);   /* lanes 0, 1 */
    float32x2_t hi = vget_high_f32(zeroed);  /* lanes 2, 3 */
    float32x2_t sum = vpadd_f32(lo, hi);     /* [0+1, 2+3] */
    return vget_lane_f32(sum, 0) + vget_lane_f32(sum, 1);
}

float reduce_max_f32(SveVecF32 v, SvePred pg) {
    /*
     * Set inactive lanes to -infinity so they can't win the max.
     */
    uint32x4_t mask = make_neon_mask(pg);
    SveVecF32 neg_inf_vec = vdupq_n_f32(-INFINITY);
    SveVecF32 masked = vbslq_f32(mask, v, neg_inf_vec);

    /* Pairwise max reduction */
    float32x2_t lo  = vget_low_f32(masked);
    float32x2_t hi  = vget_high_f32(masked);
    float32x2_t mx  = vpmax_f32(lo, hi);      /* [max(0,1), max(2,3)] */
    return fmaxf(vget_lane_f32(mx, 0), vget_lane_f32(mx, 1));
}

/* -------------------------------------------------------------------------
 * Gather / Scatter
 * ------------------------------------------------------------------------- */

SveVecF32 gather_f32(const float* base, const int32_t* indices, SvePred pg) {
    /*
     * NEON has no gather instruction. Must load each active lane individually.
     * Inactive lanes load 0.0f.
     *
     * Real SVE would be: svld1_gather_s32index_f32(pg, base, svld1_s32(pg, indices))
     */
    float buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        if (pg.active(i)) {
            buf[i] = base[indices[i]];
        }
    }
    return vld1q_f32(buf);
}

void scatter_f32(float* base, const int32_t* indices, SveVecF32 v, SvePred pg) {
    /*
     * Write each active lane to its target address.
     * Real SVE would be: svst1_scatter_s32index_f32(pg, base, svld1_s32(pg, indices), v)
     */
    float buf[4];
    vst1q_f32(buf, v);
    for (int i = 0; i < 4; ++i) {
        if (pg.active(i)) {
            base[indices[i]] = buf[i];
        }
    }
}

/* -------------------------------------------------------------------------
 * Vectorized loop helper
 * ------------------------------------------------------------------------- */

void vecop_f32(float* out, const float* a, const float* b,
               size_t len, SveOp op) {
    /*
     * This is the core SVE-style loop pattern:
     *   for each vector-width chunk (including the tail via predication)
     *     pg = whilelt(i, len)  — active only for valid elements
     *     va = predicated load
     *     vb = predicated load
     *     vc = predicated op
     *     predicated store
     *
     * The key insight: we never write a separate scalar tail loop.
     * pred_while_lt handles the tail automatically — at i=len-1 with len=5,
     * pred_while_lt(4, 5) activates only lane 0, so only one element is processed.
     */
    for (size_t i = 0; i < len; i += 4) {
        SvePred pg = pred_while_lt(i, len);

        SveVecF32 va = load_f32(a + i, pg);
        SveVecF32 vb = load_f32(b + i, pg);
        SveVecF32 vc;

        switch (op) {
            case SveOp::ADD: vc = add_f32(va, vb, pg); break;
            case SveOp::SUB: vc = sub_f32(va, vb, pg); break;
            case SveOp::MUL: vc = mul_f32(va, vb, pg); break;
            default:         vc = va; break; /* no-op for unknown op */
        }

        store_f32(out + i, vc, pg);
    }
}

} // namespace sve
} // namespace navexa
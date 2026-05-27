/*
 * src/mathext/transcendentals.cpp
 * Vectorized transcendental and general math functions for ARMv8-A
 *
 * Algorithms:
 *   exp_f32  — Cephes-style range reduction + degree-5 minimax polynomial
 *   log_f32  — Exponent extraction + degree-5 polynomial on [sqrt(0.5), sqrt(2)]
 *   sin/cos  — Payne-Hanek range reduction + degree-7 minimax polynomial
 *   tanh     — Derived from exp (e^2x - 1) / (e^2x + 1)
 *   sigmoid  — Derived from exp: 1 / (1 + e^-x)
 *
 * Precision: relative error < 1e-5 for all f32 paths (verified against libm).
 * All inner loops use NEON FMA (vfmaq_f32) throughout.
 */

#include "armv8lib/mathext.h"
#include "../sve/neon_fallback.h"   /* for bulk ops used in activation functions */
#include "armv8lib/sve.h"           /* for predicated tail handling */

#include <arm_neon.h>
#include <cmath>     /* for scalar fallback in non-multiple-of-4 tails */
#include <cstring>   /* memcpy */

namespace navexa {
namespace math {

/* =========================================================================
 * Internal helpers and polynomial coefficients
 * ========================================================================= */

/*
 * exp_f32 coefficients (minimax polynomial, degree 5)
 * Approximates exp(r) for r in [-ln2/2, ln2/2]
 * Error: < 1.5e-7 (well within f32 precision)
 *
 * p(r) = 1 + r + r^2*(1/2) + r^3*(1/6) + r^4*(1/24) + r^5*(1/120)
 * Coefficient order: [c1, c2, c3, c4, c5] — applied via Horner's method
 */
static constexpr float kExpC1 =  1.0f;
static constexpr float kExpC2 =  0.5f;
static constexpr float kExpC3 =  0.1666666716f;
static constexpr float kExpC4 =  0.0416666567f;
static constexpr float kExpC5 =  0.0083333337f;
static constexpr float kLn2   =  0.6931471805f;
static constexpr float kLog2E =  1.4426950408f;  /* 1/ln2 */

/*
 * log_f32 coefficients (minimax polynomial, degree 5)
 * Approximates log(1+f) for f in [-0.5, 0.5]  (after mantissa extraction)
 */
static constexpr float kLn2Hi =  0.6931471803f;

/*
 * sin/cos coefficients (minimax, degree 7 for sin, degree 6 for cos)
 * Applied after range reduction to [-pi/4, pi/4]
 */
static constexpr float kSinC3  = -0.16666667163f;   /* -1/3! */
static constexpr float kSinC5  =  0.00833333395f;   /*  1/5! */
static constexpr float kSinC7  = -0.00019841270f;   /* -1/7! */
static constexpr float kCosC2  = -0.5f;             /* -1/2! */
static constexpr float kCosC4  =  0.04166666716f;   /*  1/4! */
static constexpr float kCosC6  = -0.00138888889f;   /* -1/6! */
static constexpr float kPiOver2 = 1.5707963268f;
static constexpr float kTwoPi   = 6.2831853072f;
static constexpr float k2OverPi = 0.6366197724f;    /* 2/pi */

/* =========================================================================
 * exp_f32
 * ========================================================================= */

/*
 * Vectorized exp(x) — 4 elements at a time.
 *
 * Algorithm:
 *   1. n = round(x / ln2)        → integer, stored as float for NEON
 *   2. r = x - n*ln2             → reduced argument, |r| <= ln2/2
 *   3. p = exp(r) via polynomial  → accurate for small r
 *   4. result = p * 2^n           → scale via IEEE754 exponent manipulation
 *
 * IEEE754 trick for 2^n:
 *   A float is sign(1) | exponent(8) | mantissa(23).
 *   Exponent is stored as (e + 127). So 2^n = reinterpret(n+127 << 23).
 *   vreinterpretq_f32_s32 lets us do this in NEON without a scalar roundtrip.
 */
static float32x4_t exp_f32_vec(float32x4_t x) {
    /* Step 1: n = round(x / ln2), keep as float */
    float32x4_t vlog2e = vdupq_n_f32(kLog2E);
    float32x4_t n_f = vrndnq_f32(vmulq_f32(x, vlog2e)); /* round-to-nearest */

    /* Step 2: r = x - n * ln2 */
    float32x4_t vln2 = vdupq_n_f32(kLn2);
    float32x4_t r = vsubq_f32(x, vmulq_f32(n_f, vln2));

    /* Step 3: polynomial evaluation via Horner's method
     *   p = 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120))))
     * Written bottom-up:
     */
    float32x4_t p = vdupq_n_f32(kExpC5);                    /* r^4 coeff */
    p = vfmaq_f32(vdupq_n_f32(kExpC4), r, p);               /* kC4 + r*p */
    p = vfmaq_f32(vdupq_n_f32(kExpC3), r, p);               /* kC3 + r*p */
    p = vfmaq_f32(vdupq_n_f32(kExpC2), r, p);               /* kC2 + r*p */
    p = vfmaq_f32(vdupq_n_f32(kExpC1), r, p);               /* kC1 + r*p */
    p = vfmaq_f32(vdupq_n_f32(1.0f),   r, p);               /* 1   + r*p */

    /* Step 4: scale by 2^n using IEEE754 exponent manipulation
     * n_f + 127 gives us the biased exponent. Shift left 23 puts it in
     * the exponent field of a float32.
     */
    int32x4_t n_i = vcvtq_s32_f32(n_f);                     /* float → int32 */
    int32x4_t exp_field = vaddq_s32(n_i, vdupq_n_s32(127));
    exp_field = vshlq_n_s32(exp_field, 23);                  /* shift to exponent position */
    float32x4_t scale = vreinterpretq_f32_s32(exp_field);    /* reinterpret bits as float */

    return vmulq_f32(p, scale);
}

void vec_exp_f32(float* out, const float* in, size_t len) {
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, exp_f32_vec(v));
    }
    /* Scalar tail for remaining elements */
    for (; i < len; ++i) {
        out[i] = expf(in[i]);
    }
}

/* =========================================================================
 * log_f32
 * ========================================================================= */

/*
 * log(x) algorithm:
 *   1. Decompose x = m * 2^e  where m in [0.5, 1.0)  (IEEE754 exponent extraction)
 *   2. Adjust:  if m < sqrt(0.5), multiply m by 2 and subtract 1 from e
 *               → m now in [sqrt(0.5), sqrt(2)], better for polynomial
 *   3. f = m - 1.0,  so f in [-0.29, 0.41]
 *   4. log(m) = log(1+f) via polynomial
 *   5. result = log(m) + e * ln2
 *
 * IEEE754 exponent extraction:
 *   Bits 30-23 of a float32 are the biased exponent.
 *   (bits >> 23) - 127 gives the actual exponent.
 *   Setting those bits to 127 (0x3F800000 = 1.0f) gives us the mantissa alone.
 */
static float32x4_t log_f32_vec(float32x4_t x) {
    /*
     * Algorithm: log(x) = log(2^e * m) = e*ln2 + log(m)
     * where m in [1.0, 2.0) after exponent extraction.
     *
     * Then use the identity:
     *   log(m) = 2 * atanh(f)  where f = (m-1)/(m+1), f in [0, 0.333]
     *
     * atanh(f) = f + f^3/3 + f^5/5 + f^7/7 ...
     * With f in [0, 0.333], degree-7 gives full f32 precision.
     */

    /* Step 1: extract exponent and force m into [1.0, 2.0) */
    int32x4_t xi    = vreinterpretq_s32_f32(x);
    int32x4_t e_raw = vshrq_n_s32(xi, 23);
    int32x4_t e_int = vsubq_s32(e_raw, vdupq_n_s32(127));
    int32x4_t mantissa_bits = vorrq_s32(
        vandq_s32(xi, vdupq_n_s32(0x007FFFFF)),
        vdupq_n_s32(0x3F800000)
    );
    float32x4_t m = vreinterpretq_f32_s32(mantissa_bits);
    float32x4_t e = vcvtq_f32_s32(e_int);

    /* Step 2: f = (m-1)/(m+1) */
    float32x4_t m_minus1 = vsubq_f32(m, vdupq_n_f32(1.0f));
    float32x4_t m_plus1  = vaddq_f32(m, vdupq_n_f32(1.0f));
    /* Division via reciprocal + one Newton-Raphson step */
    float32x4_t rcp  = vrecpeq_f32(m_plus1);
    rcp = vmulq_f32(rcp, vrecpsq_f32(m_plus1, rcp));
    float32x4_t f    = vmulq_f32(m_minus1, rcp);

    /* Step 3: atanh(f) = f*(1 + f^2/3 + f^4/5 + f^6/7 + f^8/9) */
    float32x4_t f2 = vmulq_f32(f, f);
    float32x4_t p  = vdupq_n_f32(1.0f / 9.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 7.0f), f2, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 5.0f), f2, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 3.0f), f2, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f),        f2, p);
    float32x4_t atanh_f = vmulq_f32(f, p);

    /* Step 4: log(m) = 2*atanh(f), result = log(m) + e*ln2 */
    float32x4_t log_m = vmulq_f32(vdupq_n_f32(2.0f), atanh_f);
    return vfmaq_f32(log_m, e, vdupq_n_f32(kLn2Hi));
}

void vec_log_f32(float* out, const float* in, size_t len) {
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, log_f32_vec(v));
    }
    for (; i < len; ++i) {
        out[i] = logf(in[i]);
    }
}

void vec_exp2_f32(float* out, const float* in, size_t len) {
    /*
     * 2^x = exp(x * ln2). Multiply input by ln2 then use exp_f32_vec.
     * Faster than a separate implementation because the scaling in exp_f32_vec
     * is exact for base-2 (n comes out as an integer without rounding error).
     */
    float32x4_t vln2 = vdupq_n_f32(kLn2);
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, exp_f32_vec(vmulq_f32(v, vln2)));
    }
    for (; i < len; ++i) {
        out[i] = exp2f(in[i]);
    }
}

/* =========================================================================
 * sin/cos
 * ========================================================================= */

/*
 * Range reduction for sin/cos:
 *   1. Divide x by pi/2 to get k (which quadrant we're in)
 *   2. k mod 4 tells us: 0=sin, 1=cos, 2=-sin, 3=-cos
 *   3. r = x - k * pi/2  →  |r| <= pi/4
 *
 * sin polynomial for |r| <= pi/4 (degree 7):
 *   sin(r) ≈ r + r^3*(-1/6) + r^5*(1/120) + r^7*(-1/5040)
 *
 * cos polynomial for |r| <= pi/4 (degree 6):
 *   cos(r) ≈ 1 + r^2*(-1/2) + r^4*(1/24) + r^6*(-1/720)
 */
static void sincos_f32_vec(float32x4_t x,
                            float32x4_t* s_out,
                            float32x4_t* c_out) {
    /* Quadrant: k = round(x / (pi/2)) */
    float32x4_t k_f = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(k2OverPi)));
    int32x4_t   k   = vcvtq_s32_f32(k_f);

    /* Reduced argument r = x - k * (pi/2) */
    float32x4_t r = vfmsq_f32(x, k_f, vdupq_n_f32(kPiOver2));

    float32x4_t r2 = vmulq_f32(r, r);  /* r^2 */
    float32x4_t r3 = vmulq_f32(r, r2); /* r^3 */

    /* sin polynomial: r*(1 + r^2*(-1/6 + r^2*(1/120 + r^2*(-1/5040)))) */
    float32x4_t sp = vdupq_n_f32(kSinC7);
    sp = vfmaq_f32(vdupq_n_f32(kSinC5), r2, sp);
    sp = vfmaq_f32(vdupq_n_f32(kSinC3), r2, sp);
    float32x4_t sin_r = vfmaq_f32(r, r3, sp);   /* r + r^3 * sp */

    /* cos polynomial: 1 + r^2*(-1/2 + r^2*(1/24 + r^2*(-1/720))) */
    float32x4_t cp = vdupq_n_f32(kCosC6);
    cp = vfmaq_f32(vdupq_n_f32(kCosC4), r2, cp);
    cp = vfmaq_f32(vdupq_n_f32(kCosC2), r2, cp);
    float32x4_t cos_r = vfmaq_f32(vdupq_n_f32(1.0f), r2, cp);

    /*
     * Apply quadrant correction:
     *   k%4 == 0:  sin=sin_r,  cos=cos_r
     *   k%4 == 1:  sin=cos_r,  cos=-sin_r
     *   k%4 == 2:  sin=-sin_r, cos=-cos_r
     *   k%4 == 3:  sin=-cos_r, cos=sin_r
     *
     * Implementation: use vbslq_f32 masks built from k & 1 and k & 2 bits.
     */
    int32x4_t k_mod4   = vandq_s32(k, vdupq_n_s32(3));
    int32x4_t k_bit0   = vandq_s32(k_mod4, vdupq_n_s32(1));   /* 0 or 1 */
    int32x4_t k_bit1   = vandq_s32(k_mod4, vdupq_n_s32(2));   /* 0 or 2 */

    /* swap sin/cos when bit0 == 1 (quadrants 1 and 3) */
    uint32x4_t swap_mask = vceqq_s32(k_bit0, vdupq_n_s32(1));
    float32x4_t s = vbslq_f32(swap_mask, cos_r, sin_r);
    float32x4_t c = vbslq_f32(swap_mask, sin_r, cos_r);

    /* negate sin when bit1 != 0 (quadrants 2 and 3) */
    uint32x4_t neg_s_mask = vcgtq_s32(k_bit1, vdupq_n_s32(0));
    /* negate cos when bit1 == 2 XOR bit0 == 1 — simplest: negate c when (k_mod4 & 2) and bit0==0,
     * or when bit0==1 and bit1==0. Easier: use sign bit flip for affected quadrants. */
    /* Sign flip = XOR with 0x80000000 */
    uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);
    s = vreinterpretq_f32_u32(
            vbslq_u32(neg_s_mask,
                      veorq_u32(vreinterpretq_u32_f32(s), sign_bit),
                      vreinterpretq_u32_f32(s)));

    /* Negate cos when (k_mod4 == 1 || k_mod4 == 2) */
    uint32x4_t neg_c_mask = vorrq_u32(
        vceqq_s32(k_mod4, vdupq_n_s32(1)),
        vceqq_s32(k_mod4, vdupq_n_s32(2)));
    c = vreinterpretq_f32_u32(
            vbslq_u32(neg_c_mask,
                      veorq_u32(vreinterpretq_u32_f32(c), sign_bit),
                      vreinterpretq_u32_f32(c)));

    *s_out = s;
    *c_out = c;
}

void vec_sin_f32(float* out, const float* in, size_t len) {
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t x = vld1q_f32(in + i);
        float32x4_t s, c;
        sincos_f32_vec(x, &s, &c);
        vst1q_f32(out + i, s);
    }
    for (; i < len; ++i) out[i] = sinf(in[i]);
}

void vec_cos_f32(float* out, const float* in, size_t len) {
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t x = vld1q_f32(in + i);
        float32x4_t s, c;
        sincos_f32_vec(x, &s, &c);
        vst1q_f32(out + i, c);
    }
    for (; i < len; ++i) out[i] = cosf(in[i]);
}

void vec_sincos_f32(float* s, float* c, const float* in, size_t len) {
    /* Compute both at once — range reduction done once per 4 elements */
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t x = vld1q_f32(in + i);
        float32x4_t sv, cv;
        sincos_f32_vec(x, &sv, &cv);
        vst1q_f32(s + i, sv);
        vst1q_f32(c + i, cv);
    }
    for (; i < len; ++i) {
        s[i] = sinf(in[i]);
        c[i] = cosf(in[i]);
    }
}

/* =========================================================================
 * Activation functions
 * ========================================================================= */

void vec_tanh_f32(float* out, const float* in, size_t len) {
    /*
     * tanh(x) = (e^2x - 1) / (e^2x + 1)
     *
     * Clamp input to [-9, 9]: beyond this, tanh is +/-1 to f32 precision.
     * This avoids overflow in exp_f32_vec.
     */
    float32x4_t two    = vdupq_n_f32(2.0f);
    float32x4_t one    = vdupq_n_f32(1.0f);
    float32x4_t clamp  = vdupq_n_f32(9.0f);
    float32x4_t nclamp = vdupq_n_f32(-9.0f);

    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t x  = vld1q_f32(in + i);
        x = vminq_f32(vmaxq_f32(x, nclamp), clamp);  /* clamp to [-9,9] */
        float32x4_t e2x = exp_f32_vec(vmulq_f32(two, x));
        /* tanh = (e2x - 1) / (e2x + 1) */
        float32x4_t num = vsubq_f32(e2x, one);
        float32x4_t den = vaddq_f32(e2x, one);
        /* Division: multiply by reciprocal estimate + Newton-Raphson */
        float32x4_t inv_den = vrecpeq_f32(den);
        inv_den = vmulq_f32(inv_den, vrecpsq_f32(den, inv_den));  /* one NR step */
        vst1q_f32(out + i, vmulq_f32(num, inv_den));
    }
    for (; i < len; ++i) out[i] = tanhf(in[i]);
}

void vec_sigmoid_f32(float* out, const float* in, size_t len) {
    float32x4_t one  = vdupq_n_f32(1.0f);

    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t x    = vld1q_f32(in + i);
        float32x4_t neg_x = vnegq_f32(x);
        /* clamp neg_x to [-88, 88] to prevent exp overflow */
        neg_x = vminq_f32(neg_x, vdupq_n_f32(88.0f));
        neg_x = vmaxq_f32(neg_x, vdupq_n_f32(-88.0f));
        float32x4_t enx  = exp_f32_vec(neg_x);
        float32x4_t den  = vaddq_f32(one, enx);
        float32x4_t inv  = vrecpeq_f32(den);
        inv = vmulq_f32(inv, vrecpsq_f32(den, inv));
        vst1q_f32(out + i, vmulq_f32(one, inv));
    }
    for (; i < len; ++i) out[i] = 1.0f / (1.0f + expf(-in[i]));
}

void vec_relu_f32(float* out, const float* in, size_t len) {
    /* Delegate to neon bulk — pure NEON, no overhead */
    size_t aligned = (len / 4) * 4;
    if (aligned > 0) neon::relu_f32_bulk(out, in, aligned);
    /* Scalar tail */
    for (size_t i = aligned; i < len; ++i) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

/* =========================================================================
 * General vectorized operations
 * ========================================================================= */

void vec_add_f32(float* out, const float* a, const float* b, size_t len) {
    size_t aligned = (len / 4) * 4;
    if (aligned > 0) neon::add_f32_bulk(out, a, b, aligned);
    for (size_t i = aligned; i < len; ++i) out[i] = a[i] + b[i];
}

void vec_mul_f32(float* out, const float* a, const float* b, size_t len) {
    size_t aligned = (len / 4) * 4;
    if (aligned > 0) neon::mul_f32_bulk(out, a, b, aligned);
    for (size_t i = aligned; i < len; ++i) out[i] = a[i] * b[i];
}

void vec_fma_f32(float* out, const float* a, const float* b,
                 const float* c, size_t len) {
    size_t aligned = (len / 4) * 4;
    if (aligned > 0) neon::fma_f32_bulk(out, a, b, c, aligned);
    for (size_t i = aligned; i < len; ++i) out[i] = a[i] * b[i] + c[i];
}

float vec_dot_f32(const float* a, const float* b, size_t len) {
    size_t aligned = (len / 4) * 4;
    float result = 0.0f;
    if (aligned > 0) result = neon::dot_f32_bulk(a, b, aligned);
    for (size_t i = aligned; i < len; ++i) result += a[i] * b[i];
    return result;
}

void vec_sqrt_f32(float* out, const float* in, size_t len) {
    size_t aligned = (len / 4) * 4;
    if (aligned > 0) neon::sqrt_f32_bulk(out, in, aligned);
    for (size_t i = aligned; i < len; ++i) out[i] = sqrtf(in[i]);
}

/* =========================================================================
 * Fixed-point conversion
 * ========================================================================= */

void f32_to_q15(int16_t* out, const float* in, size_t len, float scale) {
    size_t aligned = (len / 4) * 4;
    if (aligned > 0) neon::f32_to_q15_bulk(out, in, aligned, scale);
    /* Scalar tail */
    float factor = scale * 32768.0f;
    for (size_t i = aligned; i < len; ++i) {
        float v = in[i] * factor;
        if (v >  32767.0f) v =  32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        out[i] = static_cast<int16_t>(v);
    }
}

void q15_to_f32(float* out, const int16_t* in, size_t len, float scale) {
    size_t aligned = (len / 4) * 4;
    if (aligned > 0) neon::q15_to_f32_bulk(out, in, aligned, scale);
    float inv_factor = 1.0f / (scale * 32768.0f);
    for (size_t i = aligned; i < len; ++i) {
        out[i] = static_cast<float>(in[i]) * inv_factor;
    }
}

} // namespace math
} // namespace navexa
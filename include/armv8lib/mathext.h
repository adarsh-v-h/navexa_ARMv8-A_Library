#pragma once

/*
 * navexa/mathext.h
 * Vectorized transcendental and general math functions for ARMv8-A
 *
 * All functions operate on arrays (batch/vectorized). For scalar math use <cmath>.
 * Precision target: relative error < 1e-5 for all f32 paths.
 *
 * Implementation uses:
 *   - NEON intrinsics (arm_neon.h) for all vectorized paths
 *   - Minimax polynomial approximations for transcendentals
 *   - FMA (vfmaq_f32) throughout for accuracy and speed
 *
 * Dependency: this module uses navexa::sve internally for predicated loops.
 * Callers do not need to include sve.h directly.
 */

#include <stddef.h>
#include <stdint.h>

namespace navexa {
namespace math {

/* -------------------------------------------------------------------------
 * Exponential and logarithm
 * ------------------------------------------------------------------------- */

/**
 * @brief Vectorized e^x for each element.
 *
 * Algorithm: Cephes-style range reduction + degree-5 minimax polynomial.
 * Step 1: n = round(x / ln2),  r = x - n*ln2   so |r| <= ln2/2
 * Step 2: exp(r) ≈ polynomial in r
 * Step 3: result = exp(r) * 2^n  (via float bit manipulation)
 *
 * Domain: x in [-87, 88] for f32. Values outside clamp to 0 or +inf.
 *
 * @param out  Output array. Length >= len.
 * @param in   Input array. Length >= len.
 * @param len  Number of elements to process.
 */
void vec_exp_f32(float* out, const float* in, size_t len);

/**
 * @brief Vectorized natural log ln(x) for each element.
 *
 * Algorithm: exponent extraction + degree-5 minimax polynomial on [sqrt(0.5), sqrt(2)].
 * Step 1: decompose x = m * 2^e  (extract IEEE754 exponent and mantissa)
 * Step 2: log(m) via polynomial
 * Step 3: result = log(m) + e*ln2
 *
 * Domain: x > 0. x <= 0 produces -inf or NaN matching libm behaviour.
 *
 * @param out  Output array.
 * @param in   Input array.
 * @param len  Number of elements.
 */
void vec_log_f32(float* out, const float* in, size_t len);

/**
 * @brief Vectorized 2^x for each element.
 *
 * Faster than vec_exp_f32 for base-2 because the scaling step is exact.
 * Useful for computing softmax denominators and ML attention scores.
 *
 * @param out  Output array.
 * @param in   Input array.
 * @param len  Number of elements.
 */
void vec_exp2_f32(float* out, const float* in, size_t len);

/* -------------------------------------------------------------------------
 * Trigonometry
 * ------------------------------------------------------------------------- */

/**
 * @brief Vectorized sin(x) in radians for each element.
 *
 * Algorithm: Payne-Hanek range reduction + degree-7 minimax polynomial.
 * Accurate for |x| up to ~30000 radians. Beyond that, range reduction
 * loses bits (same limitation as libm sinf).
 *
 * @param out  Output array.
 * @param in   Input array (radians).
 * @param len  Number of elements.
 */
void vec_sin_f32(float* out, const float* in, size_t len);

/**
 * @brief Vectorized cos(x) in radians for each element.
 *
 * Same algorithm as vec_sin_f32, evaluates cosine polynomial.
 *
 * @param out  Output array.
 * @param in   Input array (radians).
 * @param len  Number of elements.
 */
void vec_cos_f32(float* out, const float* in, size_t len);

/**
 * @brief Compute sin and cos simultaneously for each element.
 *
 * Faster than calling vec_sin_f32 + vec_cos_f32 separately because
 * range reduction is done once and both polynomials share it.
 * Use this when you need both values (e.g., rotation matrices).
 *
 * @param s    Output array for sin values. Length >= len.
 * @param c    Output array for cos values. Length >= len.
 * @param in   Input array (radians).
 * @param len  Number of elements.
 */
void vec_sincos_f32(float* s, float* c, const float* in, size_t len);

/* -------------------------------------------------------------------------
 * Activation functions (AI/ML)
 * ------------------------------------------------------------------------- */

/**
 * @brief Vectorized tanh(x) for each element.
 *
 * tanh(x) = (e^2x - 1) / (e^2x + 1)
 * Computed via vec_exp_f32 internally. Clamped to [-1, 1] for |x| > 9.
 * Common activation in RNNs, LSTMs.
 *
 * @param out  Output array.
 * @param in   Input array.
 * @param len  Number of elements.
 */
void vec_tanh_f32(float* out, const float* in, size_t len);

/**
 * @brief Vectorized sigmoid: 1 / (1 + e^(-x)) for each element.
 *
 * Output range (0, 1). Uses vec_exp_f32 internally.
 * Common logistic activation in binary classification layers.
 *
 * @param out  Output array.
 * @param in   Input array.
 * @param len  Number of elements.
 */
void vec_sigmoid_f32(float* out, const float* in, size_t len);

/**
 * @brief Vectorized ReLU: max(0, x) for each element.
 *
 * Implemented with vmaxq_f32(v, zero) — single NEON instruction, branchless.
 * Most common activation function in CNNs.
 *
 * @param out  Output array.
 * @param in   Input array.
 * @param len  Number of elements.
 */
void vec_relu_f32(float* out, const float* in, size_t len);

/* -------------------------------------------------------------------------
 * General vectorized operations
 * ------------------------------------------------------------------------- */

/**
 * @brief Element-wise add: out[i] = a[i] + b[i]
 *
 * @param out  Output array.
 * @param a    First input.
 * @param b    Second input.
 * @param len  Number of elements.
 */
void vec_add_f32(float* out, const float* a, const float* b, size_t len);

/**
 * @brief Element-wise multiply: out[i] = a[i] * b[i]
 */
void vec_mul_f32(float* out, const float* a, const float* b, size_t len);

/**
 * @brief Fused multiply-add: out[i] = a[i]*b[i] + c[i]
 *
 * Single NEON vfmaq_f32 instruction. No intermediate rounding.
 * Use for dot products, polynomial evaluation, convolutions.
 */
void vec_fma_f32(float* out, const float* a, const float* b,
                 const float* c, size_t len);

/**
 * @brief Dot product of two arrays.
 *
 * Returns sum of a[i]*b[i] for i in [0, len).
 * Uses pairwise NEON reduction to minimise floating-point error.
 *
 * @param a    First array.
 * @param b    Second array.
 * @param len  Number of elements. Must be >= 1.
 * @return     Scalar dot product.
 */
float vec_dot_f32(const float* a, const float* b, size_t len);

/**
 * @brief Element-wise square root: out[i] = sqrt(in[i])
 *
 * Uses vrsqrteq_f32 (reciprocal square root estimate) + two Newton-Raphson
 * refinement steps for full single-precision accuracy.
 * ~4x faster than scalar sqrtf() in a loop.
 *
 * @param out  Output array.
 * @param in   Input array. All values must be >= 0.
 * @param len  Number of elements.
 */
void vec_sqrt_f32(float* out, const float* in, size_t len);

/* -------------------------------------------------------------------------
 * Fixed-point conversion helpers
 * ------------------------------------------------------------------------- */

/**
 * @brief Convert float array to Q1.15 fixed-point (int16).
 *
 * Q1.15 format: 1 sign bit, 0 integer bits, 15 fractional bits.
 * Range: [-1.0, 1.0 - 2^-15]. Values outside this range clamp.
 *
 * Formula: out[i] = (int16_t)(in[i] * scale * 32768)
 *
 * Used for interfacing with DSP peripherals and compressed weight storage.
 *
 * @param out    Output int16 array.
 * @param in     Input float array. Values should be in [-1/scale, 1/scale].
 * @param len    Number of elements.
 * @param scale  Scaling factor applied before conversion.
 */
void f32_to_q15(int16_t* out, const float* in, size_t len, float scale);

/**
 * @brief Convert Q1.15 fixed-point (int16) back to float array.
 *
 * Formula: out[i] = (float)(in[i]) / (32768 * scale)
 *
 * @param out    Output float array.
 * @param in     Input int16 Q1.15 array.
 * @param len    Number of elements.
 * @param scale  Same scale factor used during f32_to_q15.
 */
void q15_to_f32(float* out, const int16_t* in, size_t len, float scale);

} // namespace math
} // namespace navexa
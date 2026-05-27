/*
 * tests/unit/test_mathext.cpp
 * Unit tests for navexa::math — vectorized transcendentals and general ops
 *
 * Precision tolerance for transcendentals: 1e-4 relative error
 * (our target is 1e-5, tested against libm reference values)
 *
 * What we test:
 *   - exp_f32: known values, boundary behaviour
 *   - log_f32: inverse of exp, domain edge cases
 *   - sin/cos: known angles, sincos consistency
 *   - tanh/sigmoid/relu: activation function outputs
 *   - vec_add/mul/fma/dot: correctness + non-multiple-of-4 lengths
 *   - vec_sqrt: Newton-Raphson accuracy
 *   - f32_to_q15 / q15_to_f32: round-trip consistency
 */

#include <gtest/gtest.h>
#include "armv8lib/mathext.h"

#include <cmath>
#include <vector>
#include <cstring>

using namespace navexa::math;

/* -------------------------------------------------------------------------
 * Tolerance helpers
 * ------------------------------------------------------------------------- */

/* Absolute tolerance for values expected near zero */
static constexpr float kAbsTol = 1e-4f;

/* Relative tolerance for transcendentals */
static constexpr float kRelTol = 1e-4f;

static void expect_near_rel(float actual, float expected, float rel_tol,
                             const char* msg = "") {
    float tol = rel_tol * fabsf(expected);
    if (tol < kAbsTol) tol = kAbsTol;  /* floor for values near 0 */
    EXPECT_NEAR(actual, expected, tol) << msg;
}

/* Apply expect_near_rel to each element of two arrays */
static void expect_arrays_near(const float* actual, const float* expected,
                                size_t len, float tol = kRelTol) {
    for (size_t i = 0; i < len; ++i) {
        SCOPED_TRACE("element " + std::to_string(i));
        expect_near_rel(actual[i], expected[i], tol);
    }
}

/* =========================================================================
 * vec_exp_f32 tests
 * ========================================================================= */

TEST(MathExp, KnownValues_ExactMultipleOf4) {
    float in[4]  = {0.0f, 1.0f, -1.0f, 2.0f};
    float out[4] = {};
    vec_exp_f32(out, in, 4);
    /* Compare against libm expf references */
    expect_near_rel(out[0], expf(0.0f),  kRelTol, "exp(0)");
    expect_near_rel(out[1], expf(1.0f),  kRelTol, "exp(1)");
    expect_near_rel(out[2], expf(-1.0f), kRelTol, "exp(-1)");
    expect_near_rel(out[3], expf(2.0f),  kRelTol, "exp(2)");
}

TEST(MathExp, IdentityAtZero) {
    /* exp(0) must be exactly 1.0 (or indistinguishably close) */
    float in[1]  = {0.0f};
    float out[1] = {};
    vec_exp_f32(out, in, 1);
    EXPECT_NEAR(out[0], 1.0f, kAbsTol);
}

TEST(MathExp, NonMultipleOf4_TailHandled) {
    /* len=5: 4 via NEON + 1 scalar tail */
    float in[5]  = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    float out[5] = {};
    vec_exp_f32(out, in, 5);
    for (int i = 0; i < 5; ++i) {
        SCOPED_TRACE("i=" + std::to_string(i));
        expect_near_rel(out[i], expf(in[i]), kRelTol);
    }
}

TEST(MathExp, NegativeInputs) {
    float in[4]  = {-1.0f, -2.0f, -5.0f, -10.0f};
    float out[4] = {};
    vec_exp_f32(out, in, 4);
    for (int i = 0; i < 4; ++i) {
        expect_near_rel(out[i], expf(in[i]), kRelTol);
    }
}

/* =========================================================================
 * vec_log_f32 tests
 * ========================================================================= */

TEST(MathLog, LogIsInverseOfExp) {
    /* log(exp(x)) should recover x */
    float vals[4] = {0.5f, 1.0f, 2.0f, 7.389f}; /* last = e^2 approx */
    float exp_out[4] = {};
    float log_out[4] = {};
    vec_exp_f32(exp_out, vals, 4);
    vec_log_f32(log_out, exp_out, 4);
    for (int i = 0; i < 4; ++i) {
        SCOPED_TRACE("i=" + std::to_string(i));
        EXPECT_NEAR(log_out[i], vals[i], 1e-3f);
    }
}

TEST(MathLog, LogOf1IsZero) {
    float in[1]  = {1.0f};
    float out[1] = {};
    vec_log_f32(out, in, 1);
    EXPECT_NEAR(out[0], 0.0f, kAbsTol);
}

TEST(MathLog, LogKnownValues) {
    float in[4]  = {1.0f, 2.0f, expf(1.0f), 10.0f};
    float out[4] = {};
    vec_log_f32(out, in, 4);
    expect_near_rel(out[0], 0.0f,       kRelTol, "log(1)");
    expect_near_rel(out[1], logf(2.0f), kRelTol, "log(2)");
    expect_near_rel(out[2], 1.0f,       kRelTol, "log(e)");
    expect_near_rel(out[3], logf(10.0f),kRelTol, "log(10)");
}

/* =========================================================================
 * vec_sin_f32 and vec_cos_f32 tests
 * ========================================================================= */

TEST(MathSin, KnownAngles) {
    float in[4]  = {0.0f, 3.14159f/6.0f, 3.14159f/4.0f, 3.14159f/2.0f};
    float out[4] = {};
    vec_sin_f32(out, in, 4);
    expect_near_rel(out[0], 0.0f,  kRelTol, "sin(0)");
    expect_near_rel(out[1], 0.5f,  kRelTol, "sin(pi/6)");
    expect_near_rel(out[2], 0.7071f, 1e-3f, "sin(pi/4)");
    expect_near_rel(out[3], 1.0f,  1e-3f,  "sin(pi/2)");
}

TEST(MathSin, NegativeAngles) {
    float in[4]  = {-1.0f, -2.0f, -3.0f, -0.5f};
    float out[4] = {};
    vec_sin_f32(out, in, 4);
    for (int i = 0; i < 4; ++i) {
        SCOPED_TRACE("i=" + std::to_string(i));
        expect_near_rel(out[i], sinf(in[i]), 1e-3f);
    }
}

TEST(MathCos, KnownAngles) {
    float in[4]  = {0.0f, 3.14159f/3.0f, 3.14159f/2.0f, 3.14159f};
    float out[4] = {};
    vec_cos_f32(out, in, 4);
    expect_near_rel(out[0],  1.0f,  kRelTol, "cos(0)");
    expect_near_rel(out[1],  0.5f,  1e-3f,  "cos(pi/3)");
    EXPECT_NEAR(out[2], 0.0f, 1e-3f);      /* cos(pi/2) ≈ 0 */
    expect_near_rel(out[3], -1.0f, 1e-3f, "cos(pi)");
}

TEST(MathSinCos, SincosMatchesSeparateCalls) {
    /* sincos output must exactly match calling sin and cos separately */
    float in[4]  = {0.1f, 0.5f, 1.0f, 2.5f};
    float s[4]   = {};
    float c[4]   = {};
    float s_ref[4] = {};
    float c_ref[4] = {};
    vec_sincos_f32(s, c, in, 4);
    vec_sin_f32(s_ref, in, 4);
    vec_cos_f32(c_ref, in, 4);
    for (int i = 0; i < 4; ++i) {
        SCOPED_TRACE("i=" + std::to_string(i));
        EXPECT_NEAR(s[i], s_ref[i], 1e-5f) << "sin mismatch";
        EXPECT_NEAR(c[i], c_ref[i], 1e-5f) << "cos mismatch";
    }
}

TEST(MathSinCos, PythagoreanIdentity) {
    /* sin^2(x) + cos^2(x) must equal 1 for all inputs */
    float in[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.14f};
    float s[8] = {}, c[8] = {};
    vec_sincos_f32(s, c, in, 8);
    for (int i = 0; i < 8; ++i) {
        float identity = s[i]*s[i] + c[i]*c[i];
        EXPECT_NEAR(identity, 1.0f, 1e-4f) << "Pythagorean identity failed at i=" << i;
    }
}

/* =========================================================================
 * Activation function tests
 * ========================================================================= */

TEST(MathTanh, OutputBoundedInMinusOneToOne) {
    float in[4]  = {-100.0f, -1.0f, 1.0f, 100.0f};
    float out[4] = {};
    vec_tanh_f32(out, in, 4);
    EXPECT_NEAR(out[0], -1.0f, 1e-4f);
    expect_near_rel(out[1], tanhf(-1.0f), kRelTol);
    expect_near_rel(out[2], tanhf(1.0f),  kRelTol);
    EXPECT_NEAR(out[3], 1.0f, 1e-4f);
}

TEST(MathTanh, ZeroInputGivesZero) {
    float in[1]  = {0.0f};
    float out[1] = {};
    vec_tanh_f32(out, in, 1);
    EXPECT_NEAR(out[0], 0.0f, kAbsTol);
}

TEST(MathSigmoid, OutputBoundedInZeroToOne) {
    float in[4]  = {-100.0f, 0.0f, 1.0f, 100.0f};
    float out[4] = {};
    vec_sigmoid_f32(out, in, 4);
    EXPECT_NEAR(out[0], 0.0f, 1e-4f);           /* sigmoid(-inf) = 0 */
    EXPECT_NEAR(out[1], 0.5f, kAbsTol);          /* sigmoid(0) = 0.5 */
    expect_near_rel(out[2], 1.0f/(1.0f+expf(-1.0f)), kRelTol);
    EXPECT_NEAR(out[3], 1.0f, 1e-4f);           /* sigmoid(+inf) = 1 */
}

TEST(MathRelu, NegativeInputBecomesZero) {
    float in[4]  = {-5.0f, -0.001f, 0.0f, 3.0f};
    float out[4] = {};
    vec_relu_f32(out, in, 4);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 3.0f);
}

TEST(MathRelu, PositiveInputUnchanged) {
    float in[4]  = {1.0f, 2.5f, 100.0f, 0.0001f};
    float out[4] = {};
    vec_relu_f32(out, in, 4);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.5f);
    EXPECT_FLOAT_EQ(out[2], 100.0f);
    EXPECT_FLOAT_EQ(out[3], 0.0001f);
}

/* =========================================================================
 * General ops tests
 * ========================================================================= */

TEST(MathVecAdd, BasicAddition) {
    float a[4]   = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4]   = {10.0f, 20.0f, 30.0f, 40.0f};
    float out[4] = {};
    vec_add_f32(out, a, b, 4);
    EXPECT_FLOAT_EQ(out[0], 11.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 33.0f);
    EXPECT_FLOAT_EQ(out[3], 44.0f);
}

TEST(MathVecAdd, NonMultipleOf4) {
    float a[3]   = {1.0f, 2.0f, 3.0f};
    float b[3]   = {4.0f, 5.0f, 6.0f};
    float out[3] = {};
    vec_add_f32(out, a, b, 3);
    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], 7.0f);
    EXPECT_FLOAT_EQ(out[2], 9.0f);
}

TEST(MathFma, FmaIsATimesB_PlusC) {
    float a[4]   = {2.0f, 3.0f, 4.0f, 5.0f};
    float b[4]   = {3.0f, 3.0f, 3.0f, 3.0f};
    float c[4]   = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4] = {};
    vec_fma_f32(out, a, b, c, 4);
    EXPECT_FLOAT_EQ(out[0], 7.0f);   /* 2*3+1 */
    EXPECT_FLOAT_EQ(out[1], 10.0f);  /* 3*3+1 */
    EXPECT_FLOAT_EQ(out[2], 13.0f);  /* 4*3+1 */
    EXPECT_FLOAT_EQ(out[3], 16.0f);  /* 5*3+1 */
}

TEST(MathDot, OrthogonalVectors_DotIsZero) {
    float a[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float b[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    EXPECT_NEAR(vec_dot_f32(a, b, 4), 0.0f, kAbsTol);
}

TEST(MathDot, KnownResult) {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {4.0f, 3.0f, 2.0f, 1.0f};
    /* 1*4 + 2*3 + 3*2 + 4*1 = 4+6+6+4 = 20 */
    EXPECT_NEAR(vec_dot_f32(a, b, 4), 20.0f, kAbsTol);
}

TEST(MathDot, NonMultipleOf4) {
    float a[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    /* dot = 1+2+3+4+5 = 15 */
    EXPECT_NEAR(vec_dot_f32(a, b, 5), 15.0f, kAbsTol);
}

TEST(MathSqrt, PerfectSquares) {
    float in[4]  = {1.0f, 4.0f, 9.0f, 16.0f};
    float out[4] = {};
    vec_sqrt_f32(out, in, 4);
    EXPECT_NEAR(out[0], 1.0f, 1e-4f);
    EXPECT_NEAR(out[1], 2.0f, 1e-4f);
    EXPECT_NEAR(out[2], 3.0f, 1e-4f);
    EXPECT_NEAR(out[3], 4.0f, 1e-4f);
}

TEST(MathSqrt, ArbitraryValues) {
    float in[4]  = {2.0f, 3.0f, 5.0f, 7.0f};
    float out[4] = {};
    vec_sqrt_f32(out, in, 4);
    for (int i = 0; i < 4; ++i) {
        SCOPED_TRACE("i=" + std::to_string(i));
        EXPECT_NEAR(out[i], sqrtf(in[i]), 1e-4f);
    }
}

/* =========================================================================
 * Fixed-point conversion tests
 * ========================================================================= */

TEST(MathQ15, RoundTrip_RecoverOriginalValues) {
    /*
     * Convert float → Q1.15 → float.
     * Should recover the original values within Q1.15 precision (~3e-5).
     * Using scale=1.0: input range is [-1, 1].
     */
    float in[4]  = {0.0f, 0.5f, -0.5f, 0.9f};
    int16_t q[4] = {};
    float out[4] = {};
    f32_to_q15(q, in, 4, 1.0f);
    q15_to_f32(out, q, 4, 1.0f);
    for (int i = 0; i < 4; ++i) {
        SCOPED_TRACE("i=" + std::to_string(i));
        /* Q1.15 has 1/32768 ≈ 3e-5 resolution — that's the round-trip error floor */
        EXPECT_NEAR(out[i], in[i], 3e-4f);
    }
}

TEST(MathQ15, ClampsToInt16Range) {
    /* Values outside [-1,1] with scale=1 must clamp to int16 limits */
    float in[4]  = {2.0f, -2.0f, 100.0f, -100.0f};
    int16_t q[4] = {};
    f32_to_q15(q, in, 4, 1.0f);
    EXPECT_EQ(q[0],  32767);   /* clamped to INT16_MAX */
    EXPECT_EQ(q[1], -32768);  /* clamped to INT16_MIN */
    EXPECT_EQ(q[2],  32767);
    EXPECT_EQ(q[3], -32768);
}

TEST(MathQ15, ZeroRoundTrips) {
    float in[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    int16_t q[4] = {};
    float out[4] = {};
    f32_to_q15(q, in, 4, 1.0f);
    q15_to_f32(out, q, 4, 1.0f);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }
}
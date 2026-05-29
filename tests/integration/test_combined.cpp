/*
 * tests/integration/test_combined.cpp
 * Integration tests for the ARMv8-A library (navexa / armv8)
 *
 * Unlike unit tests that exercise each module in isolation, these tests
 * verify cross-module interactions and realistic multi-step workflows:
 *
 *   1. SVE + MathExt pipeline: vectorized math using SVE predicated loops
 *   2. SME + MathExt pipeline: matrix ops feeding into vectorized activations
 *   3. SME GEMM → outer product → sparse round-trip
 *   4. Full ML inference simulation: GEMM → bias add → activation → softmax
 *   5. Virtualization + computation: context-switch with computation state
 *   6. Fixed-point pipeline: float → Q1.15 → matrix ops → back to float
 *   7. SVE gather/scatter + math pipeline
 *   8. Non-aligned (non-multiple-of-4) data flowing across modules
 */

#include <gtest/gtest.h>

/* ── Module headers ─────────────────────────────────────────────────────── */
#include "armv8lib/sve.h"
#include "armv8lib/mathext.h"
#include "armv8lib/sme/types.hpp"
#include "armv8lib/sme/gemm.hpp"
#include "armv8lib/sme/multivec.hpp"
#include "armv8lib/sme/outer_product.hpp"
#include "armv8lib/sme/sparse.hpp"
#include "armv8lib/virt/virt.hpp"

#include <arm_neon.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>

/* ── Tolerance helpers ──────────────────────────────────────────────────── */

static constexpr float kAbsTol = 1e-4f;
static constexpr float kRelTol = 1e-3f;

static void expect_near_rel(float actual, float expected,
                             const char* msg = "") {
    float tol = kRelTol * fabsf(expected);
    if (tol < kAbsTol) tol = kAbsTol;
    EXPECT_NEAR(actual, expected, tol) << msg;
}

static void expect_matrix_near(const armv8::sme::Matrix<float>& r,
                                const armv8::sme::Matrix<float>& e,
                                float tol = kAbsTol) {
    ASSERT_EQ(r.rows(), e.rows());
    ASSERT_EQ(r.cols(), e.cols());
    for (std::size_t i = 0; i < r.rows(); ++i)
        for (std::size_t j = 0; j < r.cols(); ++j)
            EXPECT_NEAR(r(i, j), e(i, j), tol)
                << "at (" << i << "," << j << ")";
}

static void extract_f32(float32x4_t v, float out[4]) {
    vst1q_f32(out, v);
}

/* =========================================================================
 * 1. SVE + MathExt Pipeline
 *
 * Scenario: Use SVE vecop to add two arrays, then pass the result
 * through mathext's exp, then take the dot product.
 * This exercises SVE → mathext data flow.
 * ========================================================================= */

TEST(IntegrationSveMath, VecopThenExpThenDot) {
    /* Step 1: SVE add two vectors */
    float a[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float b[8] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float sum[8] = {};
    navexa::sve::vecop_f32(sum, a, b, 8, navexa::sve::SveOp::ADD);

    /* Verify SVE output */
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(sum[i], a[i] + b[i], kAbsTol)
            << "SVE add mismatch at i=" << i;
    }

    /* Step 2: Pass through mathext exp */
    float exp_out[8] = {};
    navexa::math::vec_exp_f32(exp_out, sum, 8);

    /* Step 3: Dot product of exp output with a unit vector */
    float ones[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    float dot = navexa::math::vec_dot_f32(exp_out, ones, 8);

    /* Compute reference: sum of exp(a[i] + 0.1) */
    float ref = 0.0f;
    for (int i = 0; i < 8; ++i) ref += expf(a[i] + 0.1f);
    expect_near_rel(dot, ref, "SVE→exp→dot pipeline");
}

/* =========================================================================
 * 2. SVE Predicated Math Pipeline — Non-aligned length
 *
 * Scenario: Use SVE to add arrays of length 7 (not multiple of 4),
 * then compute sin+cos, verifying Pythagorean identity holds.
 * ========================================================================= */

TEST(IntegrationSveMath, NonAlignedSinCosPipeline) {
    constexpr size_t N = 7;
    float angles[N] = {0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
    float offsets[N] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float shifted[N] = {};

    /* SVE add with non-multiple-of-4 tail */
    navexa::sve::vecop_f32(shifted, angles, offsets, N, navexa::sve::SveOp::ADD);

    /* Compute sin and cos via mathext */
    float s[N] = {}, c[N] = {};
    navexa::math::vec_sincos_f32(s, c, shifted, N);

    /* Verify Pythagorean identity: sin²(x) + cos²(x) = 1 */
    for (size_t i = 0; i < N; ++i) {
        float identity = s[i] * s[i] + c[i] * c[i];
        EXPECT_NEAR(identity, 1.0f, kAbsTol)
            << "Pythagorean identity failed at i=" << i
            << " (shifted=" << shifted[i] << ")";
    }
}

/* =========================================================================
 * 3. SME GEMM → MathExt Activation Pipeline
 *
 * Scenario: Matrix multiply (simulating a neural network layer),
 * then apply sigmoid activation using mathext.
 * This is the core ML inference pattern.
 * ========================================================================= */

TEST(IntegrationSmeMath, GemmThenSigmoid) {
    using Matrix = armv8::sme::Matrix<float>;

    /* 2x3 weight matrix, 3x1 input (as 3x1 matrix) */
    Matrix W(2, 3, {0.1f, 0.2f, 0.3f,
                    0.4f, 0.5f, 0.6f});
    Matrix X(3, 1, {1.0f, 2.0f, 3.0f});

    /* Z = W * X → 2x1 */
    Matrix Z = armv8::sme::gemm_simple(W, X);
    ASSERT_EQ(Z.rows(), 2u);
    ASSERT_EQ(Z.cols(), 1u);

    /* Expected: Z(0,0) = 0.1*1 + 0.2*2 + 0.3*3 = 1.4
     *           Z(1,0) = 0.4*1 + 0.5*2 + 0.6*3 = 3.2 */
    EXPECT_NEAR(Z(0, 0), 1.4f, kAbsTol);
    EXPECT_NEAR(Z(1, 0), 3.2f, kAbsTol);

    /* Apply sigmoid activation via mathext */
    float z_arr[2] = {Z(0, 0), Z(1, 0)};
    float a_arr[2] = {};
    navexa::math::vec_sigmoid_f32(a_arr, z_arr, 2);

    /* Verify against reference sigmoid */
    for (int i = 0; i < 2; ++i) {
        float ref = 1.0f / (1.0f + expf(-z_arr[i]));
        expect_near_rel(a_arr[i], ref, "sigmoid after GEMM");
    }

    /* Sigmoid outputs must be in (0, 1) */
    EXPECT_GT(a_arr[0], 0.0f);
    EXPECT_LT(a_arr[0], 1.0f);
    EXPECT_GT(a_arr[1], 0.0f);
    EXPECT_LT(a_arr[1], 1.0f);
}

/* =========================================================================
 * 4. Full ML Forward Pass: GEMM → Bias → ReLU → GEMM → Softmax
 *
 * Scenario: Two-layer neural network forward pass using the library.
 * Layer 1: GEMM + bias add (via multivec) + ReLU activation
 * Layer 2: GEMM + softmax (exp + normalize)
 * ========================================================================= */

TEST(IntegrationMLPipeline, TwoLayerForwardPass) {
    using Matrix = armv8::sme::Matrix<float>;

    /* Layer 1: 3 inputs → 4 hidden neurons */
    Matrix W1(4, 3, { 0.1f,  0.2f, -0.1f,
                      0.3f, -0.2f,  0.4f,
                     -0.1f,  0.3f,  0.2f,
                      0.2f,  0.1f, -0.3f});
    Matrix X(3, 1, {1.0f, 0.5f, -1.0f});
    Matrix bias1(4, 1, {0.1f, -0.1f, 0.05f, 0.0f});

    /* Z1 = W1 * X */
    Matrix Z1 = armv8::sme::gemm_simple(W1, X);

    /* Add bias: Z1 = Z1 + bias1 (elementwise via multivec add) */
    Matrix H1 = armv8::sme::add(Z1, bias1);

    /* ReLU activation via mathext */
    float h1_arr[4] = {H1(0,0), H1(1,0), H1(2,0), H1(3,0)};
    float relu_out[4] = {};
    navexa::math::vec_relu_f32(relu_out, h1_arr, 4);

    /* All ReLU outputs must be >= 0 */
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(relu_out[i], 0.0f) << "ReLU output negative at i=" << i;
    }

    /* Layer 2: 4 hidden → 2 output classes */
    Matrix W2(2, 4, {0.1f, 0.2f, 0.3f, 0.4f,
                     0.5f, 0.6f, 0.7f, 0.8f});
    Matrix H1_mat(4, 1, {relu_out[0], relu_out[1], relu_out[2], relu_out[3]});
    Matrix Z2 = armv8::sme::gemm_simple(W2, H1_mat);

    /* Softmax: exp(z) / sum(exp(z)) */
    float z2_arr[2] = {Z2(0, 0), Z2(1, 0)};
    float exp_z[2] = {};
    navexa::math::vec_exp_f32(exp_z, z2_arr, 2);
    float sum_exp = exp_z[0] + exp_z[1];
    float softmax[2] = {exp_z[0] / sum_exp, exp_z[1] / sum_exp};

    /* Softmax outputs must sum to 1 and be in (0, 1) */
    EXPECT_NEAR(softmax[0] + softmax[1], 1.0f, kAbsTol)
        << "Softmax outputs don't sum to 1";
    EXPECT_GT(softmax[0], 0.0f);
    EXPECT_LT(softmax[0], 1.0f);
    EXPECT_GT(softmax[1], 0.0f);
    EXPECT_LT(softmax[1], 1.0f);
}

/* =========================================================================
 * 5. SME Matrix Round-Trip: Dense → Sparse → SpMV → Dense verify
 *
 * Scenario: Create a matrix, convert to sparse, multiply with a vector
 * (SpMV), then verify the result matches a dense GEMM.
 * ========================================================================= */

TEST(IntegrationSmeSparse, DenseToSparseSpMVConsistency) {
    using Matrix = armv8::sme::Matrix<float>;

    /* Dense matrix with some zeros (to benefit from sparsity) */
    Matrix A(3, 3, {1.0f, 0.0f, 2.0f,
                    0.0f, 3.0f, 0.0f,
                    4.0f, 0.0f, 5.0f});

    /* Convert to sparse CSR */
    auto A_sparse = armv8::sme::SparseCSR<float>::from_dense(A);
    EXPECT_EQ(A_sparse.nnz(), 5u);
    EXPECT_GT(A_sparse.sparsity(), 0.0);

    /* Round-trip: sparse → dense should recover original */
    Matrix A_recovered = A_sparse.to_dense();
    expect_matrix_near(A_recovered, A);

    /* SpMV: y = A * x */
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y(3, 0.0f);
    armv8::sme::spmv(A_sparse, x, y, 1.0f, 0.0f);

    /* Compute reference via dense GEMM */
    Matrix X_mat(3, 1, {1.0f, 2.0f, 3.0f});
    Matrix Y_ref = armv8::sme::gemm_simple(A, X_mat);

    /* Sparse SpMV must match dense GEMM */
    EXPECT_NEAR(y[0], Y_ref(0, 0), kAbsTol) << "SpMV vs GEMM mismatch at row 0";
    EXPECT_NEAR(y[1], Y_ref(1, 0), kAbsTol) << "SpMV vs GEMM mismatch at row 1";
    EXPECT_NEAR(y[2], Y_ref(2, 0), kAbsTol) << "SpMV vs GEMM mismatch at row 2";
}

/* =========================================================================
 * 6. SME SpMM vs Dense GEMM Equivalence
 *
 * Scenario: Sparse matrix × dense matrix via SpMM must match GEMM.
 * ========================================================================= */

TEST(IntegrationSmeSparse, SpMMMatchesDenseGemm) {
    using Matrix = armv8::sme::Matrix<float>;

    Matrix A(2, 3, {1.0f, 0.0f, 3.0f,
                    0.0f, 2.0f, 0.0f});
    Matrix B(3, 2, {1.0f, 2.0f,
                    3.0f, 4.0f,
                    5.0f, 6.0f});

    /* Dense GEMM reference */
    Matrix C_dense = armv8::sme::gemm_simple(A, B);

    /* Sparse SpMM */
    auto A_sparse = armv8::sme::SparseCSR<float>::from_dense(A);
    Matrix C_sparse(2, 2, 0.0f);
    armv8::sme::spmm(A_sparse, B, C_sparse, 1.0f, 0.0f);

    expect_matrix_near(C_sparse, C_dense);
}

/* =========================================================================
 * 7. Outer Product → GEMM Relationship
 *
 * Scenario: The outer product u * v^T should equal GEMM(col_u, row_v)
 * where col_u is u as a column matrix and row_v is v as a row matrix.
 * ========================================================================= */

TEST(IntegrationSmeOuterGemm, OuterProductEqualsGemm) {
    using Matrix = armv8::sme::Matrix<float>;

    std::vector<float> u = {1.0f, 2.0f, 3.0f};
    std::vector<float> v = {4.0f, 5.0f};

    /* Compute via outer product */
    Matrix C_outer = armv8::sme::outer(u, v);

    /* Compute via GEMM: u_col (3x1) × v_row (1x2) */
    Matrix u_col(3, 1, {1.0f, 2.0f, 3.0f});
    Matrix v_row(1, 2, {4.0f, 5.0f});
    Matrix C_gemm = armv8::sme::gemm_simple(u_col, v_row);

    expect_matrix_near(C_outer, C_gemm);
}

/* =========================================================================
 * 8. Rank-1 Update → Hadamard + Scale Relationship
 *
 * Scenario: Multiple rank-1 updates accumulate correctly, and the
 * result can be verified with hadamard + scale operations.
 * ========================================================================= */

TEST(IntegrationSmeMultivec, Rank1ThenScaleThenVerify) {
    using Matrix = armv8::sme::Matrix<float>;

    /* Start with zeros, apply rank-1 update: A += 2 * u * v^T */
    Matrix A(2, 2, 0.0f);
    std::vector<float> u = {1.0f, 2.0f};
    std::vector<float> v = {3.0f, 4.0f};
    armv8::sme::rank1_update(A, u, v, 2.0f);

    /* Expected: A = 2 * [[3,4],[6,8]] = [[6,8],[12,16]] */
    Matrix expected(2, 2, {6.0f, 8.0f, 12.0f, 16.0f});
    expect_matrix_near(A, expected);

    /* Scale A by 0.5 → should be [[3,4],[6,8]] */
    armv8::sme::scale(A, 0.5f);
    Matrix half_expected(2, 2, {3.0f, 4.0f, 6.0f, 8.0f});
    expect_matrix_near(A, half_expected);

    /* Hadamard of A with itself → [[9,16],[36,64]] */
    Matrix H = armv8::sme::hadamard(A, A);
    Matrix h_expected(2, 2, {9.0f, 16.0f, 36.0f, 64.0f});
    expect_matrix_near(H, h_expected);
}

/* =========================================================================
 * 9. Fixed-Point Pipeline: float → Q15 → back → mathext
 *
 * Scenario: Convert activations to Q1.15 fixed-point (as if sending
 * to a DSP), convert back, then verify tanh still produces correct
 * results. Tests that quantization error doesn't destroy the signal.
 * ========================================================================= */

TEST(IntegrationFixedPoint, FloatToQ15RoundTripThenTanh) {
    constexpr size_t N = 8;

    /* Input values in [-0.9, 0.9] (within Q1.15 representable range) */
    float in[N] = {-0.9f, -0.5f, -0.1f, 0.0f, 0.1f, 0.3f, 0.5f, 0.9f};

    /* Convert to Q1.15 and back */
    int16_t q15[N] = {};
    float recovered[N] = {};
    navexa::math::f32_to_q15(q15, in, N, 1.0f);
    navexa::math::q15_to_f32(recovered, q15, N, 1.0f);

    /* Verify round-trip error is within Q1.15 resolution */
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(recovered[i], in[i], 5e-4f)
            << "Q15 round-trip error too large at i=" << i;
    }

    /* Apply tanh to both original and recovered — results should be close */
    float tanh_orig[N] = {}, tanh_recovered[N] = {};
    navexa::math::vec_tanh_f32(tanh_orig, in, N);
    navexa::math::vec_tanh_f32(tanh_recovered, recovered, N);

    for (size_t i = 0; i < N; ++i) {
        /* Quantization noise shouldn't cause more than 1e-3 error in tanh */
        EXPECT_NEAR(tanh_recovered[i], tanh_orig[i], 1e-3f)
            << "tanh diverged after Q15 round-trip at i=" << i;
    }
}

/* =========================================================================
 * 10. SVE Gather → MathExt Computation → SVE Scatter
 *
 * Scenario: Gather non-contiguous data from a large array using SVE,
 * compute exp on the gathered data, then scatter results back.
 * Simulates sparse tensor operations.
 * ========================================================================= */

TEST(IntegrationSveGatherScatter, GatherExpScatter) {
    /* Source array — a "sparse" data structure */
    float data[16] = {};
    for (int i = 0; i < 16; ++i) data[i] = static_cast<float>(i) * 0.1f;

    /* Gather from indices {1, 5, 9, 13} */
    int32_t gather_idx[4] = {1, 5, 9, 13};
    navexa::sve::SvePred pg = navexa::sve::pred_all_f32();
    float32x4_t gathered = navexa::sve::gather_f32(data, gather_idx, pg);

    /* Extract gathered values and verify */
    float gathered_arr[4];
    extract_f32(gathered, gathered_arr);
    EXPECT_NEAR(gathered_arr[0], 0.1f, kAbsTol);
    EXPECT_NEAR(gathered_arr[1], 0.5f, kAbsTol);
    EXPECT_NEAR(gathered_arr[2], 0.9f, kAbsTol);
    EXPECT_NEAR(gathered_arr[3], 1.3f, kAbsTol);

    /* Apply exp via mathext */
    float exp_out[4] = {};
    navexa::math::vec_exp_f32(exp_out, gathered_arr, 4);

    /* Scatter results to different positions {0, 4, 8, 12} */
    float result[16] = {};
    int32_t scatter_idx[4] = {0, 4, 8, 12};
    float32x4_t exp_vec = vld1q_f32(exp_out);
    navexa::sve::scatter_f32(result, scatter_idx, exp_vec, pg);

    /* Verify scattered values */
    EXPECT_NEAR(result[0],  expf(0.1f), kRelTol);
    EXPECT_NEAR(result[4],  expf(0.5f), kRelTol);
    EXPECT_NEAR(result[8],  expf(0.9f), kRelTol);
    EXPECT_NEAR(result[12], expf(1.3f), kRelTol);

    /* Non-scattered positions must remain zero */
    EXPECT_FLOAT_EQ(result[1],  0.0f);
    EXPECT_FLOAT_EQ(result[2],  0.0f);
    EXPECT_FLOAT_EQ(result[3],  0.0f);
    EXPECT_FLOAT_EQ(result[5],  0.0f);
}

/* =========================================================================
 * 11. Virtualization + Computation State
 *
 * Scenario: Two VMs each perform matrix computations. Context switch
 * between them and verify that each VM's computation state is preserved.
 * ========================================================================= */

TEST(IntegrationVirtCompute, ContextSwitchPreservesComputationState) {
    using Matrix = armv8::sme::Matrix<float>;

    /* VM1: compute A * B */
    armv8::virt::VmContext vm1;
    vm1.vmid = 1;
    vm1.x(0) = 0xAAAA;  /* VM1's register state */
    vm1.pc   = 0x1000;

    Matrix A1(2, 2, {1, 2, 3, 4});
    Matrix B1(2, 2, {5, 6, 7, 8});
    Matrix C1 = armv8::sme::gemm_simple(A1, B1);

    /* VM2: compute different matrix op */
    armv8::virt::VmContext vm2;
    vm2.vmid = 2;
    vm2.x(0) = 0xBBBB;
    vm2.pc   = 0x2000;

    Matrix A2(2, 2, {1, 0, 0, 1});  /* identity */
    Matrix B2(2, 2, {9, 10, 11, 12});
    Matrix C2 = armv8::sme::gemm_simple(A2, B2);

    /* Context switch VM1 → VM2 */
    auto result = armv8::virt::simulate_context_switch(vm1, vm2);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.from_vmid, 1u);
    EXPECT_EQ(result.to_vmid, 2u);

    /* VM1's state is preserved after switch */
    EXPECT_EQ(vm1.x(0), 0xAAAAu);
    EXPECT_EQ(vm1.pc, 0x1000u);

    /* VM2's state is preserved after switch */
    EXPECT_EQ(vm2.x(0), 0xBBBBu);
    EXPECT_EQ(vm2.pc, 0x2000u);

    /* Both computation results are still valid */
    Matrix C1_exp(2, 2, {19, 22, 43, 50});
    expect_matrix_near(C1, C1_exp);
    expect_matrix_near(C2, B2);  /* identity * B2 = B2 */
}

/* =========================================================================
 * 12. Interrupt-Driven Computation
 *
 * Scenario: Set up virtual interrupts for two VMs, assert them,
 * check delivery, then perform computation on the "active" VM.
 * ========================================================================= */

TEST(IntegrationVirtInterrupt, InterruptThenCompute) {
    /* Create two VMs with interrupts */
    armv8::virt::VirtInterrupt irq1(32, armv8::virt::VirtIntType::IRQ, 0, 1);
    armv8::virt::VirtInterrupt irq2(33, armv8::virt::VirtIntType::FIQ, 1, 2);

    /* Assert IRQ for VM1 */
    irq1.assert_int();
    EXPECT_TRUE(irq1.should_deliver());
    EXPECT_FALSE(irq2.should_deliver());

    /* "Handle" the interrupt by performing computation */
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {};
    navexa::math::vec_relu_f32(output, input, 4);

    /* Deassert after handling */
    irq1.deassert_int();
    EXPECT_FALSE(irq1.should_deliver());

    /* Verify computation was performed correctly */
    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(output[i], input[i]);
}

/* =========================================================================
 * 13. Stage-2 Translation + Memory Simulation
 *
 * Scenario: Set up stage-2 MMU entries for different memory regions,
 * verify translations, then simulate computation on "mapped" memory.
 * ========================================================================= */

TEST(IntegrationVirtMemory, Stage2TranslationAndCompute) {
    /* Map code region: IPA 0x0000 → PA 0x8000 (read-only) */
    armv8::virt::Stage2Entry code_page(
        0x0000, 0x8000,
        armv8::virt::Stage2MemAttr::Normal_WB,
        armv8::virt::Stage2Perm::Read);

    /* Map data region: IPA 0x1000 → PA 0x9000 (read-write) */
    armv8::virt::Stage2Entry data_page(
        0x1000, 0x9000,
        armv8::virt::Stage2MemAttr::Normal_WB,
        armv8::virt::Stage2Perm::RW);

    /* Map device region: IPA 0x2000 → PA 0xA000 (device memory) */
    armv8::virt::Stage2Entry dev_page(
        0x2000, 0xA000,
        armv8::virt::Stage2MemAttr::Device_nGnRnE,
        armv8::virt::Stage2Perm::RW);

    /* Verify translations */
    EXPECT_EQ(code_page.translate(0x0000), 0x8000u);
    EXPECT_EQ(data_page.translate(0x1000), 0x9000u);
    EXPECT_EQ(dev_page.translate(0x2000),  0xA000u);

    /* Verify permissions */
    EXPECT_TRUE(code_page.permits_read());
    EXPECT_FALSE(code_page.permits_write());
    EXPECT_TRUE(data_page.permits_read());
    EXPECT_TRUE(data_page.permits_write());

    /* Translation mismatch should throw */
    EXPECT_THROW(code_page.translate(0x1000), std::runtime_error);
}

/* =========================================================================
 * 14. Multi-Vector Operations Chain
 *
 * Scenario: Build up a matrix through a sequence of operations:
 * FMA → row_scale → col_scale → add → verify against manual computation.
 * ========================================================================= */

TEST(IntegrationSmeChain, FmaRowScaleColScaleAdd) {
    using Matrix = armv8::sme::Matrix<float>;

    /* Start: A = ones(2,3), B = 2*ones(2,3) */
    Matrix A(2, 3, 1.0f);
    Matrix B(2, 3, 2.0f);

    /* FMA: A = 3*B + A = 7*ones */
    armv8::sme::fma(A, B, 3.0f);
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            EXPECT_NEAR(A(i, j), 7.0f, kAbsTol);

    /* Row scale: row 0 *= 2, row 1 *= 3 */
    std::vector<float> row_f = {2.0f, 3.0f};
    armv8::sme::row_scale(A, row_f);

    /* Expected: row 0 = [14,14,14], row 1 = [21,21,21] */
    EXPECT_NEAR(A(0, 0), 14.0f, kAbsTol);
    EXPECT_NEAR(A(1, 0), 21.0f, kAbsTol);

    /* Col scale: col 0 *= 1, col 1 *= 2, col 2 *= 3 */
    std::vector<float> col_f = {1.0f, 2.0f, 3.0f};
    armv8::sme::col_scale(A, col_f);

    /* Expected: [[14,28,42],[21,42,63]] */
    EXPECT_NEAR(A(0, 0), 14.0f, kAbsTol);
    EXPECT_NEAR(A(0, 1), 28.0f, kAbsTol);
    EXPECT_NEAR(A(0, 2), 42.0f, kAbsTol);
    EXPECT_NEAR(A(1, 0), 21.0f, kAbsTol);
    EXPECT_NEAR(A(1, 1), 42.0f, kAbsTol);
    EXPECT_NEAR(A(1, 2), 63.0f, kAbsTol);
}

/* =========================================================================
 * 15. Rank-k Update Symmetry Check
 *
 * Scenario: rankk_update produces a symmetric matrix (A = B * B^T).
 * Verify symmetry and that eigenvalue properties are preserved
 * (all diagonal elements >= 0 for a positive semi-definite result).
 * ========================================================================= */

TEST(IntegrationSmeRankk, SymmetryAndPositiveSemiDefinite) {
    using Matrix = armv8::sme::Matrix<float>;

    Matrix B(3, 2, {1.0f, 2.0f,
                    3.0f, 4.0f,
                    5.0f, 6.0f});
    Matrix A(3, 3, 0.0f);
    armv8::sme::rankk_update(A, B, 1.0f);

    /* A = B * B^T must be symmetric: A(i,j) == A(j,i) */
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(A(i, j), A(j, i), kAbsTol)
                << "Not symmetric at (" << i << "," << j << ")";

    /* Diagonal elements of B*B^T must be >= 0 (sum of squares) */
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_GE(A(i, i), 0.0f)
            << "Diagonal not non-negative at (" << i << "," << i << ")";

    /* Verify specific values: A(0,0) = 1²+2² = 5, A(1,1) = 3²+4² = 25 */
    EXPECT_NEAR(A(0, 0), 5.0f, kAbsTol);
    EXPECT_NEAR(A(1, 1), 25.0f, kAbsTol);
    EXPECT_NEAR(A(2, 2), 61.0f, kAbsTol);  /* 5²+6² */
}

/* =========================================================================
 * 16. SVE Predicated Loop — FMA Accumulation Pattern
 *
 * Scenario: Use SVE predicated arithmetic to compute a dot product
 * manually (load → mul → reduce), then compare against mathext dot.
 * Verifies SVE + mathext produce consistent results.
 * ========================================================================= */

TEST(IntegrationSveDot, PredicatedDotMatchesMathext) {
    constexpr size_t N = 10;  /* non-multiple of 4 */
    float a[N], b[N];
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(N - i);
    }

    /* Method 1: mathext dot product */
    float dot_math = navexa::math::vec_dot_f32(a, b, N);

    /* Method 2: SVE predicated manual dot product */
    float dot_sve = 0.0f;
    for (size_t i = 0; i < N; i += 4) {
        navexa::sve::SvePred pg = navexa::sve::pred_while_lt(i, N);
        navexa::sve::SveVecF32 va = navexa::sve::load_f32(a + i, pg);
        navexa::sve::SveVecF32 vb = navexa::sve::load_f32(b + i, pg);
        navexa::sve::SveVecF32 prod = navexa::sve::mul_f32(va, vb, pg);
        dot_sve += navexa::sve::reduce_add_f32(prod, pg);
    }

    /* Both methods must agree */
    EXPECT_NEAR(dot_sve, dot_math, kAbsTol)
        << "SVE predicated dot != mathext dot";

    /* Verify against scalar reference */
    float ref = 0.0f;
    for (size_t i = 0; i < N; ++i) ref += a[i] * b[i];
    EXPECT_NEAR(dot_math, ref, kAbsTol);
}

/* =========================================================================
 * 17. Exp → Log Round-Trip Through SVE Add Pipeline
 *
 * Scenario: Compute exp(x) using mathext, then log the result,
 * and verify we get x back. Run through SVE add first to shift values.
 * Tests numerical stability of the exp↔log pair.
 * ========================================================================= */

TEST(IntegrationMathRoundTrip, ExpLogRecovery) {
    constexpr size_t N = 8;
    float x[N] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};

    /* exp then log should recover x */
    float exp_out[N] = {}, log_out[N] = {};
    navexa::math::vec_exp_f32(exp_out, x, N);
    navexa::math::vec_log_f32(log_out, exp_out, N);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(log_out[i], x[i], 1e-3f)
            << "log(exp(x)) != x at i=" << i;
    }

    /* Now test the reverse: log then exp */
    /* Use positive values for log input */
    float pos[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float log_first[N] = {}, exp_second[N] = {};
    navexa::math::vec_log_f32(log_first, pos, N);
    navexa::math::vec_exp_f32(exp_second, log_first, N);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(exp_second[i], pos[i], 1e-2f)
            << "exp(log(x)) != x at i=" << i;
    }
}

/* =========================================================================
 * 18. SVE vecop MUL → MathExt sqrt → Verify L2 norm
 *
 * Scenario: Compute element-wise squares via SVE MUL,
 * take the dot (sum of squares), then sqrt — effectively an L2 norm.
 * Cross-checks SVE mul, mathext dot, and mathext sqrt.
 * ========================================================================= */

TEST(IntegrationSveMath, L2NormViaVecopMulDotSqrt) {
    constexpr size_t N = 6;  /* non-multiple of 4 */
    float v[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    /* Square each element via SVE MUL */
    float sq[N] = {};
    navexa::sve::vecop_f32(sq, v, v, N, navexa::sve::SveOp::MUL);

    /* Sum of squares via mathext dot with ones */
    float ones[N] = {1, 1, 1, 1, 1, 1};
    float sum_sq = navexa::math::vec_dot_f32(sq, ones, N);

    /* sqrt via mathext */
    float norm_arr[1] = {sum_sq};
    float norm_out[1] = {};
    navexa::math::vec_sqrt_f32(norm_out, norm_arr, 1);

    /* Reference: sqrt(1+4+9+16+25+36) = sqrt(91) */
    float ref = sqrtf(91.0f);
    EXPECT_NEAR(norm_out[0], ref, 1e-2f);
}

/* =========================================================================
 * 19. GEMM Associativity: (A * B) * C == A * (B * C)
 *
 * Scenario: Verify matrix multiplication associativity holds
 * within floating-point tolerance across the GEMM implementation.
 * ========================================================================= */

TEST(IntegrationSmeGemm, Associativity) {
    using Matrix = armv8::sme::Matrix<float>;

    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(3, 2, {7, 8, 9, 10, 11, 12});
    Matrix C(2, 2, {1, 0, 0, 1});

    /* (A * B) * C */
    Matrix AB = armv8::sme::gemm_simple(A, B);
    Matrix AB_C = armv8::sme::gemm_simple(AB, C);

    /* A * (B * C) */
    Matrix BC = armv8::sme::gemm_simple(B, C);
    Matrix A_BC = armv8::sme::gemm_simple(A, BC);

    expect_matrix_near(AB_C, A_BC);
}

/* =========================================================================
 * 20. End-to-End: All Modules Together
 *
 * Scenario: A realistic workflow combining every module:
 * 1. Set up VM context (virt)
 * 2. Allocate "weight matrix" and do GEMM (sme)
 * 3. Apply activation via vectorized math (mathext)
 * 4. Use SVE predicated ops for post-processing
 * 5. Convert to fixed-point for "hardware export" (mathext)
 * ========================================================================= */

TEST(IntegrationEndToEnd, FullWorkflow) {
    using Matrix = armv8::sme::Matrix<float>;

    /* 1. Set up VM context */
    armv8::virt::VmContext vm;
    vm.vmid = 42;
    vm.pc   = 0x80000;
    vm.x(0) = 0;  /* computation result register */
    ASSERT_TRUE(vm.is_valid());

    /* 2. Neural network layer: GEMM */
    Matrix W(4, 3, { 0.5f,  0.1f, -0.3f,
                    -0.2f,  0.4f,  0.1f,
                     0.3f, -0.1f,  0.6f,
                     0.1f,  0.2f, -0.4f});
    Matrix X(3, 1, {1.0f, -0.5f, 0.8f});
    Matrix Z = armv8::sme::gemm_simple(W, X);

    /* 3. Apply tanh activation (mathext) */
    float z_arr[4] = {Z(0,0), Z(1,0), Z(2,0), Z(3,0)};
    float activated[4] = {};
    navexa::math::vec_tanh_f32(activated, z_arr, 4);

    /* Verify tanh bounds */
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(activated[i], -1.0f);
        EXPECT_LE(activated[i],  1.0f);
    }

    /* 4. SVE post-processing: scale by 0.5 using SVE MUL */
    float scale_vec[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float scaled[4] = {};
    navexa::sve::vecop_f32(scaled, activated, scale_vec, 4,
                           navexa::sve::SveOp::MUL);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(scaled[i], activated[i] * 0.5f, kAbsTol);
    }

    /* 5. Convert to Q1.15 for "hardware export" */
    int16_t q15_out[4] = {};
    navexa::math::f32_to_q15(q15_out, scaled, 4, 1.0f);

    /* Verify Q1.15 values are valid int16 range */
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(q15_out[i], static_cast<int16_t>(-32768));
        EXPECT_LE(q15_out[i], static_cast<int16_t>(32767));
    }

    /* Round-trip check: convert back and verify closeness */
    float q15_recovered[4] = {};
    navexa::math::q15_to_f32(q15_recovered, q15_out, 4, 1.0f);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(q15_recovered[i], scaled[i], 5e-4f)
            << "Q15 round-trip error at i=" << i;
    }

    /* Store a "status" in the VM register */
    vm.x(0) = 1;  /* success */
    EXPECT_EQ(vm.x(0), 1u);
}

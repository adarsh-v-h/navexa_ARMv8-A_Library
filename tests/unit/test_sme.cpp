// =============================================================================
// tests/unit/test_sme.cpp
// Full test suite for SME matrix module:
//   - Outer products & rank-k updates
//   - Multi-vector ops
//   - Sparse SpMM / SpMV
// =============================================================================

#include "armv8lib/sme/outer_product.hpp"
#include "armv8lib/sme/multivec.hpp"
#include "armv8lib/sme/sparse.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace armv8::sme;

static void expect_near(const Matrix<float>& r, const Matrix<float>& e, float tol=1e-4f){
    ASSERT_EQ(r.rows(), e.rows()); ASSERT_EQ(r.cols(), e.cols());
    for(std::size_t i=0;i<r.rows();++i)
        for(std::size_t j=0;j<r.cols();++j)
            EXPECT_NEAR(r(i,j), e(i,j), tol) << "at (" << i << "," << j << ")";
}

// =============================================================================
// OUTER PRODUCT TESTS
// =============================================================================

TEST(OuterProductTest, Basic) {
    // u = [1,2,3], v = [4,5]
    // outer = [[4,5],[8,10],[12,15]]
    std::vector<float> u = {1,2,3}, v = {4,5};
    Matrix<float> exp(3, 2, {4,5, 8,10, 12,15});
    expect_near(outer(u, v), exp);
}

TEST(OuterProductTest, Rank1Update) {
    // A = zeros(2,2), u=[1,2], v=[3,4], alpha=1
    // A += u*v^T = [[3,4],[6,8]]
    Matrix<float> A(2, 2, 0.0f);
    std::vector<float> u = {1,2}, v = {3,4};
    rank1_update(A, u, v, 1.0f);
    Matrix<float> exp(2, 2, {3,4, 6,8});
    expect_near(A, exp);
}

TEST(OuterProductTest, Rank1UpdateAlpha) {
    // alpha=2: A += 2 * u*v^T
    Matrix<float> A(2, 2, 0.0f);
    std::vector<float> u = {1,1}, v = {1,1};
    rank1_update(A, u, v, 2.0f);
    Matrix<float> exp(2, 2, {2,2, 2,2});
    expect_near(A, exp);
}

TEST(OuterProductTest, RankkUpdate) {
    // B = [[1,0],[0,1]] (identity 2x2)
    // A = zeros(2,2), rankk: A += B*B^T = I*I^T = I
    Matrix<float> A(2, 2, 0.0f);
    Matrix<float> B(2, 2, {1,0, 0,1});
    rankk_update(A, B, 1.0f);
    Matrix<float> exp(2, 2, {1,0, 0,1});
    expect_near(A, exp);
}

TEST(OuterProductTest, DimMismatch) {
    Matrix<float> A(2, 3, 0.0f);
    std::vector<float> u = {1,2,3}, v = {1,2}; // u.size != A.rows
    EXPECT_THROW(rank1_update(A, u, v), std::invalid_argument);
}

// =============================================================================
// MULTI-VECTOR OPS TESTS
// =============================================================================

TEST(MultivecTest, Scale) {
    Matrix<float> A(2, 2, {1,2,3,4});
    scale(A, 2.0f);
    Matrix<float> exp(2, 2, {2,4,6,8});
    expect_near(A, exp);
}

TEST(MultivecTest, Add) {
    Matrix<float> A(2, 2, {1,2,3,4});
    Matrix<float> B(2, 2, {4,3,2,1});
    Matrix<float> exp(2, 2, {5,5,5,5});
    expect_near(add(A, B), exp);
}

TEST(MultivecTest, Hadamard) {
    Matrix<float> A(2, 2, {1,2,3,4});
    Matrix<float> B(2, 2, {2,2,2,2});
    Matrix<float> exp(2, 2, {2,4,6,8});
    expect_near(hadamard(A, B), exp);
}

TEST(MultivecTest, FMA) {
    // A = ones, B = ones, alpha=3 → A = 3*B + A = 4*ones
    Matrix<float> A(2, 2, 1.0f);
    Matrix<float> B(2, 2, 1.0f);
    fma(A, B, 3.0f);
    Matrix<float> exp(2, 2, 4.0f);
    expect_near(A, exp);
}

TEST(MultivecTest, RowScale) {
    Matrix<float> A(3, 2, {1,1, 2,2, 3,3});
    std::vector<float> f = {1,2,3};
    row_scale(A, f);
    Matrix<float> exp(3, 2, {1,1, 4,4, 9,9});
    expect_near(A, exp);
}

TEST(MultivecTest, ColScale) {
    Matrix<float> A(2, 3, {1,2,3, 1,2,3});
    std::vector<float> f = {1,2,3};
    col_scale(A, f);
    Matrix<float> exp(2, 3, {1,4,9, 1,4,9});
    expect_near(A, exp);
}

TEST(MultivecTest, DimMismatch) {
    Matrix<float> A(2,2), B(3,3);
    EXPECT_THROW(add(A,B), std::invalid_argument);
    EXPECT_THROW(hadamard(A,B), std::invalid_argument);
    EXPECT_THROW(fma(A,B,1.0f), std::invalid_argument);
}

// =============================================================================
// SPARSE TESTS
// =============================================================================

TEST(SparseTest, FromDenseAndBack) {
    Matrix<float> dense(2, 3, {1,0,2, 0,3,0});
    auto sparse = SparseCSR<float>::from_dense(dense);
    EXPECT_EQ(sparse.nnz(), 3u);
    EXPECT_NEAR(sparse.sparsity(), 0.5, 1e-4);
    auto back = sparse.to_dense();
    expect_near(back, dense);
}

TEST(SparseTest, SpMV) {
    // A = [[1,0],[0,2]], x=[3,4] → y = [3,8]
    Matrix<float> dense(2, 2, {1,0, 0,2});
    auto A = SparseCSR<float>::from_dense(dense);
    std::vector<float> x = {3,4}, y = {0,0};
    spmv(A, x, y, 1.0f, 0.0f);
    EXPECT_NEAR(y[0], 3.0f, 1e-4f);
    EXPECT_NEAR(y[1], 8.0f, 1e-4f);
}

TEST(SparseTest, SpMM) {
    // A_sparse = identity 2x2, B = [[1,2],[3,4]] → C = B
    Matrix<float> id(2, 2, {1,0, 0,1});
    auto A = SparseCSR<float>::from_dense(id);
    Matrix<float> B(2, 2, {1,2, 3,4});
    Matrix<float> C(2, 2, 0.0f);
    spmm(A, B, C, 1.0f, 0.0f);
    expect_near(C, B);
}

TEST(SparseTest, SpMMAlphaBeta) {
    // C = 2*I*B + 0*C = 2*B
    Matrix<float> id(2, 2, {1,0, 0,1});
    auto A = SparseCSR<float>::from_dense(id);
    Matrix<float> B(2, 2, {1,2, 3,4});
    Matrix<float> C(2, 2, 0.0f);
    spmm(A, B, C, 2.0f, 0.0f);
    Matrix<float> exp(2, 2, {2,4, 6,8});
    expect_near(C, exp);
}

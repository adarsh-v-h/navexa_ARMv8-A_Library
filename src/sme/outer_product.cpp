// =============================================================================
// src/sme/outer_product.cpp
// SME Matrix Acceleration — Outer Product & Rank-k Update Implementation
// =============================================================================

#include "armv8lib/sme/outer_product.hpp"
#include <stdexcept>

namespace armv8 {
namespace sme {

// outer: C = u * v^T
template <typename T>
Matrix<T> outer(const std::vector<T>& u, const std::vector<T>& v)
{
    const std::size_t M = u.size(), N = v.size();
    if (M == 0 || N == 0)
        throw std::invalid_argument("outer: vectors must be non-empty");
    Matrix<T> C(M, N, T{0});
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
            C(i, j) = u[i] * v[j];
    return C;
}

// rank1_update: A = A + alpha * u * v^T
template <typename T>
void rank1_update(Matrix<T>& A,
                  const std::vector<T>& u,
                  const std::vector<T>& v,
                  T alpha)
{
    if (A.rows() != u.size() || A.cols() != v.size())
        throw std::invalid_argument("rank1_update: dimension mismatch");
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            A(i, j) += alpha * u[i] * v[j];
}

// rankk_update: A = A + alpha * B * B^T
template <typename T>
void rankk_update(Matrix<T>& A, const Matrix<T>& B, T alpha)
{
    if (A.rows() != B.rows() || A.cols() != B.rows())
        throw std::invalid_argument("rankk_update: A must be B.rows x B.rows");
    const std::size_t M = B.rows(), K = B.cols();
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < M; ++j)
            for (std::size_t k = 0; k < K; ++k)
                A(i, j) += alpha * B(i, k) * B(j, k);
}

// Explicit instantiations
template Matrix<float>  outer<float> (const std::vector<float>&,  const std::vector<float>&);
template Matrix<double> outer<double>(const std::vector<double>&, const std::vector<double>&);
template void rank1_update<float> (Matrix<float>&,  const std::vector<float>&,  const std::vector<float>&,  float);
template void rank1_update<double>(Matrix<double>&, const std::vector<double>&, const std::vector<double>&, double);
template void rankk_update<float> (Matrix<float>&,  const Matrix<float>&,  float);
template void rankk_update<double>(Matrix<double>&, const Matrix<double>&, double);

} // namespace sme
} // namespace armv8

// =============================================================================
// src/sme/multivec.cpp
// SME Matrix Acceleration — Multi-Vector Operations Implementation
// =============================================================================

#include "armv8lib/sme/multivec.hpp"
#include <stdexcept>

namespace armv8 {
namespace sme {

template <typename T>
void scale(Matrix<T>& A, T alpha)
{
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            A(i, j) *= alpha;
}

template <typename T>
Matrix<T> add(const Matrix<T>& A, const Matrix<T>& B)
{
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw std::invalid_argument("add: matrices must have same dimensions");
    Matrix<T> C(A.rows(), A.cols(), T{0});
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            C(i, j) = A(i, j) + B(i, j);
    return C;
}

template <typename T>
Matrix<T> hadamard(const Matrix<T>& A, const Matrix<T>& B)
{
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw std::invalid_argument("hadamard: matrices must have same dimensions");
    Matrix<T> C(A.rows(), A.cols(), T{0});
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            C(i, j) = A(i, j) * B(i, j);
    return C;
}

template <typename T>
void fma(Matrix<T>& A, const Matrix<T>& B, T alpha)
{
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw std::invalid_argument("fma: matrices must have same dimensions");
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            A(i, j) += alpha * B(i, j);
}

template <typename T>
void row_scale(Matrix<T>& A, const std::vector<T>& factors)
{
    if (factors.size() != A.rows())
        throw std::invalid_argument("row_scale: factors.size() must equal A.rows()");
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            A(i, j) *= factors[i];
}

template <typename T>
void col_scale(Matrix<T>& A, const std::vector<T>& factors)
{
    if (factors.size() != A.cols())
        throw std::invalid_argument("col_scale: factors.size() must equal A.cols()");
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            A(i, j) *= factors[j];
}

// Explicit instantiations
template void     scale<float>   (Matrix<float>&,  float);
template void     scale<double>  (Matrix<double>&, double);
template Matrix<float>  add<float>  (const Matrix<float>&,  const Matrix<float>&);
template Matrix<double> add<double> (const Matrix<double>&, const Matrix<double>&);
template Matrix<float>  hadamard<float>  (const Matrix<float>&,  const Matrix<float>&);
template Matrix<double> hadamard<double> (const Matrix<double>&, const Matrix<double>&);
template void fma<float> (Matrix<float>&,  const Matrix<float>&,  float);
template void fma<double>(Matrix<double>&, const Matrix<double>&, double);
template void row_scale<float> (Matrix<float>&,  const std::vector<float>&);
template void row_scale<double>(Matrix<double>&, const std::vector<double>&);
template void col_scale<float> (Matrix<float>&,  const std::vector<float>&);
template void col_scale<double>(Matrix<double>&, const std::vector<double>&);

} // namespace sme
} // namespace armv8

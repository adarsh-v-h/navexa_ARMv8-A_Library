#pragma once
#include "types.hpp"
namespace armv8 {
namespace sme {
template <typename T>
void gemm(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C,
          T alpha = T{1}, T beta = T{0});
template <typename T>
Matrix<T> gemm_simple(const Matrix<T>& A, const Matrix<T>& B);
} // namespace sme
} // namespace armv8

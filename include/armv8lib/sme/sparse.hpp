#pragma once
#include "types.hpp"
#include <vector>
namespace armv8 {
namespace sme {
template <typename T>
void spmm(const SparseCSR<T>& A, const Matrix<T>& B, Matrix<T>& C,
          T alpha = T{1}, T beta = T{0});
template <typename T>
void spmv(const SparseCSR<T>& A, const std::vector<T>& x, std::vector<T>& y,
          T alpha = T{1}, T beta = T{0});
} // namespace sme
} // namespace armv8

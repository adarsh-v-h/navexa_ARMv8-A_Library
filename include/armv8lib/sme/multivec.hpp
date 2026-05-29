#pragma once

// =============================================================================
// armv8lib/sme/multivec.hpp
// SME Matrix Acceleration — Multi-Vector Operations
//
// Implements vectorized elementwise ops across matrix rows/cols:
//   scale        : A = alpha * A
//   add          : C = A + B  (elementwise)
//   multiply     : C = A * B  (elementwise, Hadamard)
//   fma          : A = alpha * B + A  (fused multiply-add)
//   row_scale    : scale each row i by factors[i]
//   col_scale    : scale each col j by factors[j]
//
// Namespace : armv8::sme
// Standard  : C++17
// =============================================================================

#include "types.hpp"

namespace armv8 {
namespace sme {

// Scale all elements: A = alpha * A
template <typename T>
void scale(Matrix<T>& A, T alpha);

// Elementwise add: C = A + B
template <typename T>
Matrix<T> add(const Matrix<T>& A, const Matrix<T>& B);

// Elementwise multiply (Hadamard): C = A * B
template <typename T>
Matrix<T> hadamard(const Matrix<T>& A, const Matrix<T>& B);

// Fused multiply-add: A = alpha * B + A
template <typename T>
void fma(Matrix<T>& A, const Matrix<T>& B, T alpha);

// Scale each row i by factors[i]
template <typename T>
void row_scale(Matrix<T>& A, const std::vector<T>& factors);

// Scale each col j by factors[j]
template <typename T>
void col_scale(Matrix<T>& A, const std::vector<T>& factors);

} // namespace sme
} // namespace armv8

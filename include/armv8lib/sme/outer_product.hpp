#pragma once

// =============================================================================
// armv8lib/sme/outer_product.hpp
// SME Matrix Acceleration — Outer Products & Rank-k Updates
//
// Implements:
//   rank1_update : A = A + alpha * u * v^T   (rank-1)
//   outer        : returns u * v^T as a new matrix
//   rankk_update : A = A + alpha * B * B^T   (rank-k, symmetric)
//
// Namespace : armv8::sme
// Standard  : C++17
// =============================================================================

#include "types.hpp"

namespace armv8 {
namespace sme {

// -----------------------------------------------------------------------------
// outer()
// Computes C = u * v^T where u is (M x 1) and v is (N x 1)
// Returns a new (M x N) matrix
// -----------------------------------------------------------------------------
template <typename T>
Matrix<T> outer(const std::vector<T>& u, const std::vector<T>& v);

// -----------------------------------------------------------------------------
// rank1_update()
// Computes A = A + alpha * u * v^T  (in-place)
// A must be pre-allocated as (u.size() x v.size())
// -----------------------------------------------------------------------------
template <typename T>
void rank1_update(Matrix<T>& A,
                  const std::vector<T>& u,
                  const std::vector<T>& v,
                  T alpha = T{1});

// -----------------------------------------------------------------------------
// rankk_update()
// Computes A = A + alpha * B * B^T  (in-place, symmetric rank-k update)
// A must be square (B.rows() x B.rows())
// -----------------------------------------------------------------------------
template <typename T>
void rankk_update(Matrix<T>& A,
                  const Matrix<T>& B,
                  T alpha = T{1});

} // namespace sme
} // namespace armv8

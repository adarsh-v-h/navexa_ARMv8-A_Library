#pragma once
/*
 * src/sve/neon_fallback.h
 * Internal header — not part of the public navexa API.
 * Only included by src/sve/sve_emulation.cpp and src/mathext/transcendentals.cpp
 */

#include <stddef.h>
#include <stdint.h>

namespace navexa {
namespace neon {

/* Bulk arithmetic — len must be a multiple of 4 */
void add_f32_bulk(float* out, const float* a, const float* b, size_t len);
void sub_f32_bulk(float* out, const float* a, const float* b, size_t len);
void mul_f32_bulk(float* out, const float* a, const float* b, size_t len);
void fma_f32_bulk(float* out, const float* a, const float* b, const float* c, size_t len);

/* Reduction */
float dot_f32_bulk(const float* a, const float* b, size_t len);

/* Element-wise */
void sqrt_f32_bulk(float* out, const float* in, size_t len);
void relu_f32_bulk(float* out, const float* in, size_t len);

/* Fixed-point conversion — len must be a multiple of 4 */
void f32_to_q15_bulk(int16_t* out, const float* in, size_t len, float scale);
void q15_to_f32_bulk(float* out, const int16_t* in, size_t len, float scale);

} // namespace neon
} // namespace navexa
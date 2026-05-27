#pragma once

/*
 * navexa/sve.h
 * SVE/SVE2 emulation layer + optimized NEON fallback
 *
 * On ARMv8-A hardware (NEON only), all operations are emulated at 128-bit
 * vector width using NEON intrinsics. If real SVE hardware is detected at
 * runtime via AT_HWCAP, sve_is_hw_available() returns true — future versions
 * can dispatch to native SVE paths.
 *
 * Predication model:
 *   Real SVE uses hardware predicate registers. We emulate them with SvePred,
 *   a bitmask where bit i = 1 means lane i is active.
 *   Merging predication: inactive lanes retain their original value.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <arm_neon.h>

namespace navexa {
namespace sve {

/* -------------------------------------------------------------------------
 * Emulated SVE types
 * -------------------------------------------------------------------------
 * SveVecF32 : 4 x float32  (128-bit, matches one NEON Q-register)
 * SveVecF64 : 2 x float64  (128-bit, matches one NEON Q-register)
 * SvePred   : lane-active bitmask, bit i set = lane i is active
 *             For f32: bits 0-3 used. For f64: bits 0-1 used.
 * SveOp     : operation selector for sve_vecop_f32
 * ------------------------------------------------------------------------- */

using SveVecF32 = float32x4_t;
using SveVecF64 = float64x2_t;

struct SvePred {
    uint8_t mask; /* bit i = lane i active, LSB = lane 0 */

    /* convenience: true if lane i is active */
    bool active(int lane) const { return (mask >> lane) & 1u; }
};

enum class SveOp : uint8_t {
    ADD,  /* out[i] = a[i] + b[i] */
    SUB,  /* out[i] = a[i] - b[i] */
    MUL,  /* out[i] = a[i] * b[i] */
};

/* -------------------------------------------------------------------------
 * Detection
 * ------------------------------------------------------------------------- */

/**
 * @brief Check whether the CPU supports real SVE instructions.
 *
 * Reads AT_HWCAP2 via getauxval(). On most ARMv8-A cores this returns false.
 * The emulation layer is always available regardless of this flag.
 *
 * @return true if hardware SVE is present, false if emulation is used.
 */
bool is_hw_available();

/**
 * @brief Return the emulated vector length in bits.
 *
 * Always returns 128 in the current emulation layer (one NEON register).
 * On real SVE hardware this would return the hardware VL.
 */
size_t get_emulated_vl();

/* -------------------------------------------------------------------------
 * Predicate construction
 * ------------------------------------------------------------------------- */

/**
 * @brief Create a predicate with all 4 f32 lanes active.
 */
SvePred pred_all_f32();

/**
 * @brief Create a predicate with all 2 f64 lanes active.
 */
SvePred pred_all_f64();

/**
 * @brief Create a predicate with the first n lanes active (n <= 4 for f32).
 *
 * Equivalent to SVE whilelt starting from 0.
 * Example: pred_first_n(3) → mask = 0b0111 (lanes 0,1,2 active)
 *
 * @param n  Number of active lanes from the start. Clamped to [0,4].
 */
SvePred pred_first_n(size_t n);

/**
 * @brief Compute active lanes for a tail loop iteration.
 *
 * Returns a predicate where lane k is active if (idx + k) < limit.
 * Direct equivalent of SVE whilelt(idx, limit).
 *
 * Usage in a vectorized loop:
 *   for (size_t i = 0; i < len; i += 4) {
 *       SvePred pg = pred_while_lt(i, len);
 *       ...
 *   }
 *
 * @param idx    Current base index.
 * @param limit  Total length of the array.
 */
SvePred pred_while_lt(size_t idx, size_t limit);

/* -------------------------------------------------------------------------
 * Predicated load / store
 * ------------------------------------------------------------------------- */

/**
 * @brief Load up to 4 floats from ptr, gated by predicate pg.
 *
 * Inactive lanes are loaded as 0.0f. Safe to call at array tail where
 * fewer than 4 elements remain — just set pg via pred_while_lt.
 *
 * @param ptr  Source pointer. Must be valid for at least popcount(pg.mask) floats.
 * @param pg   Predicate controlling which lanes are loaded.
 */
SveVecF32 load_f32(const float* ptr, SvePred pg);

/**
 * @brief Store up to 4 floats to ptr, gated by predicate pg.
 *
 * Only active lanes are written. Inactive lanes leave memory unchanged.
 *
 * @param ptr  Destination pointer.
 * @param v    Vector to store.
 * @param pg   Predicate controlling which lanes are stored.
 */
void store_f32(float* ptr, SveVecF32 v, SvePred pg);

/**
 * @brief Load up to 2 doubles from ptr, gated by predicate pg.
 */
SveVecF64 load_f64(const double* ptr, SvePred pg);

/**
 * @brief Store up to 2 doubles to ptr, gated by predicate pg.
 */
void store_f64(double* ptr, SveVecF64 v, SvePred pg);

/* -------------------------------------------------------------------------
 * Predicated arithmetic (merging)
 *
 * Merging predication: active lanes get the computed result.
 * Inactive lanes retain the value of the first operand (a).
 * This matches SVE's default merging behaviour.
 * ------------------------------------------------------------------------- */

/**
 * @brief Predicated f32 add:  result[i] = pg[i] ? a[i]+b[i] : a[i]
 */
SveVecF32 add_f32(SveVecF32 a, SveVecF32 b, SvePred pg);

/**
 * @brief Predicated f32 subtract:  result[i] = pg[i] ? a[i]-b[i] : a[i]
 */
SveVecF32 sub_f32(SveVecF32 a, SveVecF32 b, SvePred pg);

/**
 * @brief Predicated f32 multiply:  result[i] = pg[i] ? a[i]*b[i] : a[i]
 */
SveVecF32 mul_f32(SveVecF32 a, SveVecF32 b, SvePred pg);

/**
 * @brief Predicated fused multiply-add:  result[i] = pg[i] ? a[i]*b[i]+c[i] : a[i]
 *
 * Uses NEON vfmaq_f32 which is a single instruction on any ARMv8-A core.
 * No rounding between the multiply and add — full FMA precision.
 */
SveVecF32 fma_f32(SveVecF32 a, SveVecF32 b, SveVecF32 c, SvePred pg);

/**
 * @brief Predicated f64 add.
 */
SveVecF64 add_f64(SveVecF64 a, SveVecF64 b, SvePred pg);

/**
 * @brief Predicated f64 multiply.
 */
SveVecF64 mul_f64(SveVecF64 a, SveVecF64 b, SvePred pg);

/* -------------------------------------------------------------------------
 * Horizontal reduction
 * ------------------------------------------------------------------------- */

/**
 * @brief Sum of active f32 lanes.
 *
 * Inactive lanes contribute 0.0f to the sum.
 *
 * @param v   Input vector.
 * @param pg  Predicate selecting active lanes.
 * @return    Sum of active lanes.
 */
float reduce_add_f32(SveVecF32 v, SvePred pg);

/**
 * @brief Maximum of active f32 lanes.
 *
 * Inactive lanes are excluded. Behaviour is undefined if no lanes are active.
 *
 * @param v   Input vector.
 * @param pg  Predicate selecting active lanes.
 * @return    Maximum value among active lanes.
 */
float reduce_max_f32(SveVecF32 v, SvePred pg);

/* -------------------------------------------------------------------------
 * Gather / Scatter (emulated — no NEON gather instruction exists)
 * ------------------------------------------------------------------------- */

/**
 * @brief Gather load: result[i] = base[indices[i]] for active lanes.
 *
 * Inactive lanes load 0.0f. Emulated with scalar loads since NEON has
 * no gather instruction. Equivalent to SVE LD1W with scalar offset.
 *
 * @param base     Base array pointer.
 * @param indices  Array of 4 int32 indices.
 * @param pg       Predicate controlling which lanes are loaded.
 */
SveVecF32 gather_f32(const float* base, const int32_t* indices, SvePred pg);

/**
 * @brief Scatter store: base[indices[i]] = v[i] for active lanes.
 *
 * Inactive lanes write nothing. Emulated with scalar stores.
 *
 * @param base     Base array pointer.
 * @param indices  Array of 4 int32 indices.
 * @param v        Values to scatter.
 * @param pg       Predicate controlling which lanes are stored.
 */
void scatter_f32(float* base, const int32_t* indices, SveVecF32 v, SvePred pg);

/* -------------------------------------------------------------------------
 * Vectorized loop helper
 * ------------------------------------------------------------------------- */

/**
 * @brief Apply a binary operation across two full arrays using predicated loops.
 *
 * This is the canonical SVE-style loop: iterate in steps of 4, using
 * pred_while_lt to handle the tail naturally — no separate scalar cleanup needed.
 *
 * Equivalent C:
 *   for (size_t i = 0; i < len; i++) out[i] = op(a[i], b[i]);
 *
 * But vectorized at 4 elements per iteration with proper tail handling.
 *
 * @param out  Output array. Length >= len.
 * @param a    First input array. Length >= len.
 * @param b    Second input array. Length >= len.
 * @param len  Number of elements to process.
 * @param op   Operation to apply (ADD, SUB, or MUL).
 */
void vecop_f32(float* out, const float* a, const float* b, size_t len, SveOp op);

} // namespace sve
} // namespace navexa
# Module Design — SVE/NEON and Math/Transcendentals
## navexa library | Person A

---

## Overview

Two modules. One depends on the other.

```
mathext (transcendentals.cpp)
    └── uses sve (neon_fallback.cpp) for its vectorized paths
    └── uses sve (sve_emulation.cpp) when SVE is detected at runtime

sve
    ├── neon_fallback.cpp   — fixed-width 128-bit NEON implementations
    └── sve_emulation.cpp   — SVE emulation layer on top of NEON
```

`sve` has no dependencies on other navexa modules.
`mathext` depends only on `sve` — nothing else.

---

## Module 1 — SVE (`include/armv8lib/sve.h`)

### What SVE is and why we emulate it

SVE (Scalable Vector Extension) is ARM's post-NEON SIMD architecture.
The key difference: SVE vectors are *length-agnostic* — hardware can be
128 to 2048 bits wide, and the same code runs on all widths using predicate
registers to mask active lanes.

ARMv8-A base has NEON (always 128-bit). No SVE. So we emulate SVE's
programming model on top of NEON: fixed 128-bit vectors, software-managed
predicate masks, scatter/gather via loops.

### Design decisions

- All SVE-style operations work on `float` (f32) and `double` (f64) for now.
  Integer SVE ops are out of scope for this version.
- Predicate = a bitmask (`uint8_t pred[16]`, one byte per lane, 0=inactive 1=active).
  This matches the spirit of SVE's `svbool_t` without requiring SVE hardware.
- Emulated vector width = 128 bits = 4×f32 or 2×f64. This is what NEON gives us.
- Gather/scatter uses scalar loops — there is no NEON gather instruction.
- At runtime, `sve_is_hw_available()` checks `AT_HWCAP` for real SVE support.
  If present, we could use real SVE intrinsics; for now we always use the
  NEON path since SVE hardware isn't our target.

### Functions

#### Detection
| Function | Signature | Description |
|---|---|---|
| `sve_is_hw_available` | `bool ()` | Check `/proc/cpuinfo` or `AT_HWCAP` for SVE |
| `sve_get_emulated_vl` | `size_t ()` | Returns emulated vector length in bits (always 128) |

#### Predicate helpers
| Function | Signature | Description |
|---|---|---|
| `sve_pred_all` | `SvePred (size_t len)` | All lanes active up to len |
| `sve_pred_first_n` | `SvePred (size_t n)` | First n lanes active, rest masked |
| `sve_pred_while_lt` | `SvePred (size_t idx, size_t limit)` | Active while idx+lane < limit (SVE whilelt equivalent) |

#### Contiguous load/store (predicated)
| Function | Signature | Description |
|---|---|---|
| `sve_load_f32` | `SveVecF32 (const float* ptr, SvePred pg)` | Load 4 floats, masked by predicate |
| `sve_store_f32` | `void (float* ptr, SveVecF32 v, SvePred pg)` | Store 4 floats, masked by predicate |
| `sve_load_f64` | `SveVecF64 (const double* ptr, SvePred pg)` | Load 2 doubles, masked |
| `sve_store_f64` | `void (double* ptr, SveVecF64 v, SvePred pg)` | Store 2 doubles, masked |

#### Arithmetic (predicated, merging)
| Function | Signature | Description |
|---|---|---|
| `sve_add_f32` | `SveVecF32 (SveVecF32 a, SveVecF32 b, SvePred pg)` | Add where active, pass-through where masked |
| `sve_sub_f32` | `SveVecF32 (SveVecF32 a, SveVecF32 b, SvePred pg)` | Subtract, predicated |
| `sve_mul_f32` | `SveVecF32 (SveVecF32 a, SveVecF32 b, SvePred pg)` | Multiply, predicated |
| `sve_fma_f32` | `SveVecF32 (SveVecF32 a, SveVecF32 b, SveVecF32 c, SvePred pg)` | `a*b + c`, predicated (fused multiply-add) |
| `sve_add_f64` | `SveVecF64 (SveVecF64 a, SveVecF64 b, SvePred pg)` | f64 add, predicated |
| `sve_mul_f64` | `SveVecF64 (SveVecF64 a, SveVecF64 b, SvePred pg)` | f64 multiply, predicated |

#### Horizontal reduction
| Function | Signature | Description |
|---|---|---|
| `sve_reduce_add_f32` | `float (SveVecF32 v, SvePred pg)` | Sum of active lanes |
| `sve_reduce_max_f32` | `float (SveVecF32 v, SvePred pg)` | Max of active lanes |

#### Gather / Scatter (emulated)
| Function | Signature | Description |
|---|---|---|
| `sve_gather_f32` | `SveVecF32 (const float* base, const int32_t* indices, SvePred pg)` | Load from non-contiguous addresses |
| `sve_scatter_f32` | `void (float* base, const int32_t* indices, SveVecF32 v, SvePred pg)` | Store to non-contiguous addresses |

#### Vectorized loop helper (the main SVE-style entry point)
| Function | Signature | Description |
|---|---|---|
| `sve_vecop_f32` | `void (float* out, const float* a, const float* b, size_t len, SveOp op)` | Apply op across full array using predicated loop — the idiomatic SVE loop emulation |

---

## Module 2 — MathExt (`include/armv8lib/mathext.h`)

### Design decisions

- All functions operate on arrays, not scalars. Scalar math already exists in `<cmath>`.
  The value of this module is batch/vectorized operations.
- All functions take `(float* out, const float* in, size_t len)` form for consistency.
- Precision target: relative error < 1e-5 for f32 paths. This is sufficient for
  AI activations and HPC simulations. Not sufficient for cryptography (don't use here).
- Implementation strategy: polynomial minimax approximations for sin/cos/exp/log.
  These map directly to NEON FMA instructions and are faster than libm.
- SVE fallback path is used when `sve_is_hw_available()` returns true.
  NEON path is always available.

### Functions

#### Exponential and logarithm
| Function | Signature | Description |
|---|---|---|
| `vec_exp_f32` | `void (float* out, const float* in, size_t len)` | e^x for each element |
| `vec_log_f32` | `void (float* out, const float* in, size_t len)` | natural log for each element |
| `vec_exp2_f32` | `void (float* out, const float* in, size_t len)` | 2^x — useful for ML, maps to ARM `vexpq` trick |

#### Trigonometry
| Function | Signature | Description |
|---|---|---|
| `vec_sin_f32` | `void (float* out, const float* in, size_t len)` | sin(x), input in radians |
| `vec_cos_f32` | `void (float* out, const float* in, size_t len)` | cos(x), input in radians |
| `vec_sincos_f32` | `void (float* s, float* c, const float* in, size_t len)` | sin and cos together (shares range reduction — faster than calling both) |

#### Activation functions (AI/ML focus)
| Function | Signature | Description |
|---|---|---|
| `vec_tanh_f32` | `void (float* out, const float* in, size_t len)` | tanh — common RNN activation |
| `vec_sigmoid_f32` | `void (float* out, const float* in, size_t len)` | 1/(1+e^-x) — logistic activation |
| `vec_relu_f32` | `void (float* out, const float* in, size_t len)` | max(0, x) — trivial but NEON makes it branchless |

#### General vectorized ops
| Function | Signature | Description |
|---|---|---|
| `vec_add_f32` | `void (float* out, const float* a, const float* b, size_t len)` | Element-wise add |
| `vec_mul_f32` | `void (float* out, const float* a, const float* b, size_t len)` | Element-wise multiply |
| `vec_fma_f32` | `void (float* out, const float* a, const float* b, const float* c, size_t len)` | out = a*b + c |
| `vec_dot_f32` | `float (const float* a, const float* b, size_t len)` | Dot product — returns scalar |
| `vec_sqrt_f32` | `void (float* out, const float* in, size_t len)` | Square root, uses `vrsqrteq` + Newton-Raphson |

#### Fixed-point helpers
| Function | Signature | Description |
|---|---|---|
| `f32_to_q15` | `void (int16_t* out, const float* in, size_t len, float scale)` | Convert float to Q1.15 fixed-point |
| `q15_to_f32` | `void (float* out, const int16_t* in, size_t len, float scale)` | Convert Q1.15 back to float |

---

## Internal Implementation Notes

### exp_f32 algorithm (Cephes minimax, adapted for NEON)
```
1. Range reduction:  x = n*ln2 + r,  where n = round(x/ln2), |r| <= ln2/2
2. Polynomial:       exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120  (degree-5)
3. Scale:            result = exp(r) * 2^n   (done via float bit manipulation)
```
All three steps vectorize cleanly with NEON FMA.

### sin_f32 algorithm (minimax polynomial)
```
1. Range reduction:  fold x into [-π/4, π/4] tracking quadrant
2. Polynomial:       sin(r) ≈ r - r³/6 + r⁵/120 - r⁷/5040  (degree-7)
                     cos(r) ≈ 1 - r²/2 + r⁴/24 - r⁶/720   (degree-6)
3. Quadrant fix:     apply sign and sin/cos swap based on tracked quadrant
```

### SVE predicate emulation
Real SVE uses hardware predicate registers that gate each lane independently.
We emulate this with a struct holding a `uint8_t mask[4]` (one byte per f32 lane).
Arithmetic functions check the mask and blend: active lanes get computed result,
inactive lanes keep their original value (merging predication).

---

## File Ownership

| File | Owner |
|---|---|
| `include/armv8lib/sve.h` | Person A |
| `include/armv8lib/mathext.h` | Person A |
| `src/sve/sve_emulation.cpp` | Person A |
| `src/sve/neon_fallback.cpp` | Person A |
| `src/mathext/transcendentals.cpp` | Person A |
| `tests/unit/test_sve.cpp` | Person A |
| `tests/unit/test_mathext.cpp` | Person A |

No other person modifies these files. If another module needs SVE or mathext,
they include the header and call through the public API.

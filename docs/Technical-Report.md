# navexa ARMv8-A Library — Technical Report

**Project:** navexa_ARMv8-A_Library  
**Repository:** [github.com/adarsh-v-h/navexa_ARMv8-A_Library](https://github.com/adarsh-v-h/navexa_ARMv8-A_Library)  
**Documentation:** [deepwiki.com/adarsh-v-h/navexa_ARMv8-A_Library](https://deepwiki.com/adarsh-v-h/navexa_ARMv8-A_Library/1-project-overview)  
**Language:** C++20  
**Target Architecture:** ARMv8-A (AArch64)  
**Build System:** CMake 3.20+ with aarch64-linux-gnu cross-compilation  
**Testing:** Google Test + QEMU User-Mode Emulation  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Project Architecture & Repository Structure](#2-project-architecture--repository-structure)
3. [Development Environment & Toolchain](#3-development-environment--toolchain)
4. [Module 1 — SVE/NEON Emulation Layer](#4-module-1--sveneon-emulation-layer)
5. [Module 2 — Vectorized Math & Transcendentals](#5-module-2--vectorized-math--transcendentals)
6. [Module 3 — SME Matrix Acceleration](#6-module-3--sme-matrix-acceleration)
7. [Module 4 — Memory Tagging Extension (MTE) Simulation](#7-module-4--memory-tagging-extension-mte-simulation)
8. [Module 5 — LSE Atomic Operations](#8-module-5--lse-atomic-operations)
9. [Module 6 — Virtualization & Hypervisor Helpers](#9-module-6--virtualization--hypervisor-helpers)
10. [Testing Strategy & Integration Tests](#10-testing-strategy--integration-tests)
11. [Performance Benchmarking](#11-performance-benchmarking)
12. [Build System & CI/CD Pipeline](#12-build-system--cicd-pipeline)
13. [Use Cases & Applications](#13-use-cases--applications)
14. [Conclusion](#14-conclusion)

---

## 1. Introduction

The **navexa ARMv8-A Library** is a production-grade C/C++ library that provides high-performance implementations of advanced ARM architectural features on the base ARMv8-A platform. The core challenge it addresses is the gap between what the ARMv8-A base specification provides (NEON 128-bit SIMD) and the capabilities introduced in later ARM extensions (SVE, SME, MTE) that most developers cannot yet rely on in production.

The library is organized into six primary modules:

| Module | Feature | Complexity |
|--------|---------|------------|
| SVE/NEON Emulation | Scalable Vector Extension emulation on top of NEON | High |
| Math/Transcendentals | Vectorized exp, log, sin, cos, tanh, sigmoid, ReLU | Medium |
| SME Matrix Acceleration | GEMM, outer products, sparse matrix ops | High |
| MTE Simulation | Memory Tagging Extension for memory safety | Medium |
| Atomics (LSE) | Lock-free data structures with ARMv8.1 LSE atomics | Medium |
| Virtualization | VM context switching and stage-2 MMU abstractions | Medium-High |

All modules compile with the AArch64 cross-compiler (`aarch64-linux-gnu-g++`) and execute on x86 development machines via QEMU user-mode emulation, requiring **no ARM hardware** for development or testing.

---

## 2. Project Architecture & Repository Structure

The library enforces strict module boundaries — each module lives in its own directory under `src/`, with public APIs exposed through headers in `include/armv8lib/`. This design ensures independent development across team members with minimal merge conflicts.

```
navexa_ARMv8-A_Library/
├── include/armv8lib/          # Public API headers
│   ├── sve.h                  # SVE emulation types and functions
│   ├── mathext.h              # Vectorized math API
│   ├── mte.h                  # Memory Tagging Extension API
│   ├── atomics.h              # LSE atomic operations
│   ├── sme.h                  # SME umbrella header
│   ├── sme/
│   │   ├── types.hpp          # Matrix<T> and SparseCSR<T> types
│   │   ├── gemm.hpp           # General Matrix Multiply
│   │   ├── outer_product.hpp  # Outer product and rank-k updates
│   │   ├── multivec.hpp       # Multi-vector matrix operations
│   │   └── sparse.hpp         # Sparse matrix operations (CSR)
│   └── virt/
│       └── virt.hpp           # Virtualization helpers
├── src/
│   ├── sve/
│   │   ├── sve_emulation.cpp  # SVE predication, gather/scatter, vecop
│   │   ├── neon_fallback.cpp  # Pure NEON bulk kernels
│   │   └── neon_fallback.h    # Internal header for NEON kernels
│   ├── mathext/
│   │   └── transcendentals.cpp # All vectorized math implementations
│   ├── sme/
│   │   ├── gemm.cpp           # Tiled GEMM implementation
│   │   ├── outer_product.cpp  # Outer product and rank-k updates
│   │   ├── multivec.cpp       # Matrix element-wise operations
│   │   └── sparse.cpp         # SpMM and SpMV operations
│   ├── mte/
│   │   ├── mte_sim.cpp        # MTE allocation and tag management
│   │   └── bounds_check.cpp   # Tag comparison and validation
│   ├── atomics/
│   │   └── atomic_ops.cpp     # FreeList push/pop with CASPAL
│   ├── rng/
│   │   └── rndr.cpp           # Random number generation
│   └── virt/
│       └── vhe_helpers.cpp    # Virtualization helpers
├── tests/
│   ├── unit/                  # Per-module unit tests
│   └── integration/
│       └── test_combined.cpp  # Cross-module integration tests
├── benchmarks/
│   └── bench_sve.cpp          # Micro-benchmark suite
├── cmake/
│   └── toolchain-aarch64.cmake # Cross-compilation toolchain
├── CMakeLists.txt             # Root build configuration
├── .clang-format              # Code style enforcement
└── .github/workflows/ci.yml  # GitHub Actions CI pipeline
```

### Design Principles

1. **Header–Source Separation:** Public headers contain only declarations and inline trivial accessors. All NEON intrinsic usage (`#include <arm_neon.h>`) is confined to `.cpp` files, keeping headers clean for consumers.

2. **Namespace Isolation:** Each module uses its own namespace — `navexa::sve`, `navexa::math`, `navexa::mte`, `armv8lib::atomic`, `armv8::sme`, `armv8::virt` — preventing symbol collisions.

3. **Static Library Output:** The entire library compiles into a single static archive (`libarmv8lib.a`), making it trivial to link into downstream projects.

---

## 3. Development Environment & Toolchain

### Cross-Compilation Stack

The library targets AArch64 but is developed entirely on x86-64 Linux. The toolchain consists of:

| Component | Purpose |
|-----------|---------|
| `aarch64-linux-gnu-g++` | Cross-compiler producing ARM binaries |
| `qemu-aarch64` | User-mode emulator running ARM binaries on x86 |
| `cmake` + `ninja-build` | Build system with cross-compilation support |
| `libgtest-dev` | Google Test framework for unit testing |
| `clang-format` / `clang-tidy` | Code style enforcement and static analysis |

### Installation

```bash
sudo apt update
sudo apt install -y \
  gcc-aarch64-linux-gnu \
  g++-aarch64-linux-gnu \
  binutils-aarch64-linux-gnu \
  qemu-user \
  qemu-user-static \
  cmake \
  ninja-build \
  git \
  clang-format \
  clang-tidy \
  libgtest-dev
```

### Cross-Compilation Toolchain File

The `cmake/toolchain-aarch64.cmake` file tells CMake to use the ARM cross-compiler:

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

### Build and Test Commands

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-aarch64.cmake \
         -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### QEMU Environment

The `QEMU_LD_PREFIX` environment variable must be set so QEMU can locate ARM shared libraries:

```bash
export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu
```

This allows cross-compiled ARM binaries to run transparently on x86 machines, enabling the entire development and test cycle without ARM hardware.

---

## 4. Module 1 — SVE/NEON Emulation Layer

### Overview

ARM's Scalable Vector Extension (SVE) introduces length-agnostic SIMD programming — the same code runs on hardware with 128-bit to 2048-bit vector widths. However, SVE is not available on the base ARMv8-A platform. This module emulates SVE's programming model using NEON's fixed 128-bit vectors, providing:

- **Predicate-based lane masking** (emulating SVE's `svbool_t`)
- **Predicated load/store** with safe tail handling
- **Predicated arithmetic** with merging semantics
- **Gather/scatter** memory access patterns
- **Horizontal reductions** (sum, max)
- **SVE-style vectorized loop helpers**

### Key Types

```cpp
// Emulated SVE vector types — aliases for NEON registers
using SveVecF32 = float32x4_t;   // 4 × float32 (128-bit)
using SveVecF64 = float64x2_t;   // 2 × float64 (128-bit)

// Predicate: bitmask where bit i = lane i active
struct SvePred {
    uint8_t mask;
    bool active(int lane) const { return (mask >> lane) & 1u; }
};

// Operation selector for vectorized loops
enum class SveOp : uint8_t { ADD, SUB, MUL };
```

### Hardware Detection

The library checks for real SVE hardware at runtime via the Linux auxiliary vector:

```cpp
bool is_hw_available() {
#if defined(__aarch64__) && defined(HWCAP2_SVE)
    return (getauxval(AT_HWCAP2) & HWCAP2_SVE) != 0;
#else
    return false;
#endif
}
```

On standard ARMv8-A hardware this returns `false`, and the emulation layer is always used.

### Predicate Construction

Predicates control which vector lanes are active during operations. Three constructors cover the common patterns:

```cpp
// All 4 lanes active (full vector)
SvePred pred_all_f32() {
    return SvePred{0x0F};  // bits 0-3 set
}

// First n lanes active (for partial operations)
SvePred pred_first_n(size_t n) {
    if (n == 0) return SvePred{0x00};
    if (n >= 4) return SvePred{0x0F};
    return SvePred{static_cast<uint8_t>((1u << n) - 1u)};
}

// SVE whilelt equivalent — active while (idx + lane) < limit
SvePred pred_while_lt(size_t idx, size_t limit) {
    uint8_t mask = 0;
    for (int k = 0; k < 4; ++k) {
        if ((idx + static_cast<size_t>(k)) < limit)
            mask |= static_cast<uint8_t>(1u << k);
    }
    return SvePred{mask};
}
```

### Predicated Load/Store

Loads and stores respect the predicate — only active lanes participate:

```cpp
SveVecF32 load_f32(const float* ptr, SvePred pg) {
    float buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        if (pg.active(i)) buf[i] = ptr[i];
    }
    return vld1q_f32(buf);
}

void store_f32(float* ptr, SveVecF32 v, SvePred pg) {
    float buf[4];
    vst1q_f32(buf, v);
    for (int i = 0; i < 4; ++i) {
        if (pg.active(i)) ptr[i] = buf[i];
    }
}
```

### Predicated Arithmetic with Merging Semantics

Active lanes get the computed result; inactive lanes retain the value of the first operand. This matches SVE's default merging predication:

```cpp
// Internal helper: expand SvePred into NEON lane mask for vbslq_f32
static uint32x4_t make_neon_mask(SvePred pg) {
    uint32_t m[4];
    for (int i = 0; i < 4; ++i)
        m[i] = pg.active(i) ? 0xFFFFFFFFu : 0x00000000u;
    return vld1q_u32(m);
}

SveVecF32 add_f32(SveVecF32 a, SveVecF32 b, SvePred pg) {
    SveVecF32 result = vaddq_f32(a, b);
    uint32x4_t mask = make_neon_mask(pg);
    return vbslq_f32(mask, result, a);  // blend: active→result, inactive→a
}

SveVecF32 fma_f32(SveVecF32 a, SveVecF32 b, SveVecF32 c, SvePred pg) {
    SveVecF32 result = vfmaq_f32(c, a, b);  // a*b + c (fused, single instruction)
    uint32x4_t mask = make_neon_mask(pg);
    return vbslq_f32(mask, result, a);
}
```

### Horizontal Reductions

```cpp
float reduce_add_f32(SveVecF32 v, SvePred pg) {
    uint32x4_t mask = make_neon_mask(pg);
    SveVecF32 zeroed = vbslq_f32(mask, v, vdupq_n_f32(0.0f));
    float32x2_t lo = vget_low_f32(zeroed);
    float32x2_t hi = vget_high_f32(zeroed);
    float32x2_t sum = vpadd_f32(lo, hi);  // pairwise add: [0+1, 2+3]
    return vget_lane_f32(sum, 0) + vget_lane_f32(sum, 1);
}
```

### Gather/Scatter (Emulated)

NEON lacks gather/scatter instructions, so these are emulated with scalar loops:

```cpp
SveVecF32 gather_f32(const float* base, const int32_t* indices, SvePred pg) {
    float buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        if (pg.active(i)) buf[i] = base[indices[i]];
    }
    return vld1q_f32(buf);
}
```

### SVE-Style Vectorized Loop Helper

The canonical SVE loop pattern — no separate scalar tail cleanup needed:

```cpp
void vecop_f32(float* out, const float* a, const float* b,
               size_t len, SveOp op) {
    for (size_t i = 0; i < len; i += 4) {
        SvePred pg = pred_while_lt(i, len);
        SveVecF32 va = load_f32(a + i, pg);
        SveVecF32 vb = load_f32(b + i, pg);
        SveVecF32 vc;
        switch (op) {
            case SveOp::ADD: vc = add_f32(va, vb, pg); break;
            case SveOp::SUB: vc = sub_f32(va, vb, pg); break;
            case SveOp::MUL: vc = mul_f32(va, vb, pg); break;
            default:         vc = va; break;
        }
        store_f32(out + i, vc, pg);
    }
}
```

---

## 5. Module 2 — Vectorized Math & Transcendentals

### Overview

This module provides batch-vectorized implementations of mathematical functions using NEON intrinsics with minimax polynomial approximations. All functions operate on arrays (not scalars) and achieve relative error < 1e-5 for f32 paths.

### Function Summary

| Category | Functions |
|----------|-----------|
| Exponential/Log | `vec_exp_f32`, `vec_log_f32`, `vec_exp2_f32` |
| Trigonometry | `vec_sin_f32`, `vec_cos_f32`, `vec_sincos_f32` |
| AI/ML Activations | `vec_tanh_f32`, `vec_sigmoid_f32`, `vec_relu_f32` |
| General Ops | `vec_add_f32`, `vec_mul_f32`, `vec_fma_f32`, `vec_dot_f32`, `vec_sqrt_f32` |
| Fixed-Point | `f32_to_q15`, `q15_to_f32` |

### Vectorized exp(x) — Cephes-Style Minimax

The `exp_f32` implementation uses a degree-5 minimax polynomial with IEEE754 bit manipulation for the 2^n scaling:

```cpp
static float32x4_t exp_f32_vec(float32x4_t x) {
    // Step 1: n = round(x / ln2)
    float32x4_t n_f = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(kLog2E)));

    // Step 2: r = x - n * ln2  (reduced argument, |r| <= ln2/2)
    float32x4_t r = vsubq_f32(x, vmulq_f32(n_f, vdupq_n_f32(kLn2)));

    // Step 3: Degree-5 polynomial via Horner's method
    float32x4_t p = vdupq_n_f32(kExpC5);
    p = vfmaq_f32(vdupq_n_f32(kExpC4), r, p);
    p = vfmaq_f32(vdupq_n_f32(kExpC3), r, p);
    p = vfmaq_f32(vdupq_n_f32(kExpC2), r, p);
    p = vfmaq_f32(vdupq_n_f32(kExpC1), r, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f),   r, p);

    // Step 4: Scale by 2^n via IEEE754 exponent manipulation
    int32x4_t n_i = vcvtq_s32_f32(n_f);
    int32x4_t exp_field = vshlq_n_s32(
        vaddq_s32(n_i, vdupq_n_s32(127)), 23);
    float32x4_t scale = vreinterpretq_f32_s32(exp_field);

    return vmulq_f32(p, scale);
}
```

The algorithm processes 4 elements per iteration. The IEEE754 trick — constructing `2^n` by shifting `(n + 127)` into the exponent field — avoids expensive `pow()` calls entirely.

### Vectorized log(x) — Exponent Extraction + Atanh Polynomial

```cpp
static float32x4_t log_f32_vec(float32x4_t x) {
    // Extract IEEE754 exponent: e = (bits >> 23) - 127
    int32x4_t xi    = vreinterpretq_s32_f32(x);
    int32x4_t e_int = vsubq_s32(vshrq_n_s32(xi, 23), vdupq_n_s32(127));

    // Force mantissa into [1.0, 2.0) by masking exponent bits
    int32x4_t mantissa_bits = vorrq_s32(
        vandq_s32(xi, vdupq_n_s32(0x007FFFFF)),
        vdupq_n_s32(0x3F800000));
    float32x4_t m = vreinterpretq_f32_s32(mantissa_bits);
    float32x4_t e = vcvtq_f32_s32(e_int);

    // f = (m-1)/(m+1), computed via reciprocal estimate + Newton-Raphson
    float32x4_t rcp = vrecpeq_f32(vaddq_f32(m, vdupq_n_f32(1.0f)));
    rcp = vmulq_f32(rcp, vrecpsq_f32(vaddq_f32(m, vdupq_n_f32(1.0f)), rcp));
    float32x4_t f = vmulq_f32(vsubq_f32(m, vdupq_n_f32(1.0f)), rcp);

    // atanh(f) via degree-9 polynomial, then log(m) = 2*atanh(f)
    float32x4_t f2 = vmulq_f32(f, f);
    float32x4_t p = vdupq_n_f32(1.0f / 9.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 7.0f), f2, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 5.0f), f2, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 3.0f), f2, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f), f2, p);

    float32x4_t log_m = vmulq_f32(vdupq_n_f32(2.0f), vmulq_f32(f, p));
    return vfmaq_f32(log_m, e, vdupq_n_f32(kLn2Hi));
}
```

### Vectorized sin/cos — Range Reduction + Quadrant Correction

The `sincos_f32_vec` function computes both sin and cos simultaneously, sharing the range reduction step:

```cpp
static void sincos_f32_vec(float32x4_t x,
                            float32x4_t* s_out, float32x4_t* c_out) {
    // Quadrant: k = round(x / (pi/2))
    float32x4_t k_f = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(k2OverPi)));
    int32x4_t   k   = vcvtq_s32_f32(k_f);

    // Reduced argument: r = x - k * (pi/2), |r| <= pi/4
    float32x4_t r = vfmsq_f32(x, k_f, vdupq_n_f32(kPiOver2));
    float32x4_t r2 = vmulq_f32(r, r);

    // Sin polynomial (degree 7): r + r^3*(-1/6 + r^2*(1/120 + r^2*(-1/5040)))
    float32x4_t sp = vdupq_n_f32(kSinC7);
    sp = vfmaq_f32(vdupq_n_f32(kSinC5), r2, sp);
    sp = vfmaq_f32(vdupq_n_f32(kSinC3), r2, sp);
    float32x4_t sin_r = vfmaq_f32(r, vmulq_f32(r, r2), sp);

    // Cos polynomial (degree 6): 1 + r^2*(-1/2 + r^2*(1/24 + r^2*(-1/720)))
    float32x4_t cp = vdupq_n_f32(kCosC6);
    cp = vfmaq_f32(vdupq_n_f32(kCosC4), r2, cp);
    cp = vfmaq_f32(vdupq_n_f32(kCosC2), r2, cp);
    float32x4_t cos_r = vfmaq_f32(vdupq_n_f32(1.0f), r2, cp);

    // Quadrant correction using NEON bitselect and sign flips
    // k%4: 0→(sin,cos), 1→(cos,-sin), 2→(-sin,-cos), 3→(-cos,sin)
    // ... (swap and negate based on quadrant bits)
}
```

### Activation Functions for AI/ML

```cpp
// tanh(x) = (e^2x - 1) / (e^2x + 1), clamped to [-9,9] to prevent overflow
void vec_tanh_f32(float* out, const float* in, size_t len) {
    float32x4_t two = vdupq_n_f32(2.0f), one = vdupq_n_f32(1.0f);
    for (size_t i = 0; i + 4 <= len; i += 4) {
        float32x4_t x  = vld1q_f32(in + i);
        x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-9.0f)), vdupq_n_f32(9.0f));
        float32x4_t e2x = exp_f32_vec(vmulq_f32(two, x));
        float32x4_t num = vsubq_f32(e2x, one);
        float32x4_t den = vaddq_f32(e2x, one);
        float32x4_t inv = vrecpeq_f32(den);
        inv = vmulq_f32(inv, vrecpsq_f32(den, inv));  // Newton-Raphson
        vst1q_f32(out + i, vmulq_f32(num, inv));
    }
}

// ReLU: max(0, x) — single NEON instruction, branchless
void vec_relu_f32(float* out, const float* in, size_t len) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < len; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, vmaxq_f32(v, zero));  // one instruction
    }
}
```

### Square Root via Newton-Raphson

Instead of computing `sqrt(x)` directly, the NEON approach computes `1/sqrt(x)` first using the hardware estimate, refines it with two Newton-Raphson steps, then multiplies by `x`:

```cpp
void sqrt_f32_bulk(float* out, const float* in, size_t len) {
    for (size_t i = 0; i < len; i += 4) {
        float32x4_t vx = vld1q_f32(in + i);
        float32x4_t e = vrsqrteq_f32(vx);        // ~8-bit estimate
        // NR step 1: ~16-bit accurate
        e = vmulq_f32(e, vrsqrtsq_f32(vx, vmulq_f32(e, e)));
        // NR step 2: ~23-bit accurate (full f32 precision)
        e = vmulq_f32(e, vrsqrtsq_f32(vx, vmulq_f32(e, e)));
        vst1q_f32(out + i, vmulq_f32(vx, e));     // x * (1/sqrt(x)) = sqrt(x)
    }
}
```

### Dot Product with Pairwise Reduction

```cpp
float dot_f32_bulk(const float* a, const float* b, size_t len) {
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < len; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vacc = vfmaq_f32(vacc, va, vb);  // fused multiply-accumulate
    }
    // Pairwise reduction: [s0+s1, s2+s3] → scalar
    float32x2_t lo  = vget_low_f32(vacc);
    float32x2_t hi  = vget_high_f32(vacc);
    float32x2_t sum = vpadd_f32(lo, hi);
    return vget_lane_f32(sum, 0) + vget_lane_f32(sum, 1);
}
```

### Fixed-Point Conversion (Q1.15)

Used for interfacing with DSP peripherals and compressed weight storage:

```cpp
void f32_to_q15_bulk(int16_t* out, const float* in, size_t len, float scale) {
    float32x4_t vscale = vdupq_n_f32(scale * 32768.0f);
    for (size_t i = 0; i < len; i += 4) {
        float32x4_t vf = vmulq_f32(vld1q_f32(in + i), vscale);
        int32x4_t vi32 = vcvtnq_s32_f32(vf);     // round-to-nearest
        int16x4_t vi16 = vqmovn_s32(vi32);        // saturating narrow
        vst1_s16(out + i, vi16);
    }
}
```

---

## 6. Module 3 — SME Matrix Acceleration

### Overview

The Scalable Matrix Extension (SME) module provides efficient matrix operations for AI inference, training, and HPC workloads. Since SME hardware is only available in Armv9, this module provides portable C++ implementations with cache-friendly tiled algorithms.

### Matrix Type

A generic `Matrix<T>` class provides the foundation for all matrix operations:

```cpp
template <typename T>
class Matrix {
    static_assert(std::is_arithmetic<T>::value, "Matrix<T>: T must be arithmetic");
public:
    Matrix(size_type r, size_type c) : rows_(r), cols_(c), data_(r*c, T{}) {}
    Matrix(size_type r, size_type c, std::initializer_list<T> v);

    T& operator()(size_type i, size_type j) noexcept { return data_[i*cols_+j]; }
    void fill(T v) noexcept { std::fill(data_.begin(), data_.end(), v); }
    void zero() noexcept { fill(T{}); }
private:
    size_type rows_, cols_;
    std::vector<T> data_;
};
```

### General Matrix Multiply (GEMM)

The GEMM implementation uses a 64×64 tile-based approach for cache efficiency:

```cpp
template <typename T>
void gemm(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C,
          T alpha, T beta) {
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    if (beta == T{0}) C.zero();
    else if (beta != T{1})
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t j = 0; j < N; ++j) C(i,j) *= beta;

    constexpr std::size_t TILE = 64;
    for (std::size_t i0 = 0; i0 < M; i0 += TILE) {
        const std::size_t ie = std::min(i0 + TILE, M);
        for (std::size_t k0 = 0; k0 < K; k0 += TILE) {
            const std::size_t ke = std::min(k0 + TILE, K);
            for (std::size_t j0 = 0; j0 < N; j0 += TILE) {
                const std::size_t je = std::min(j0 + TILE, N);
                for (std::size_t i = i0; i < ie; ++i)
                    for (std::size_t k = k0; k < ke; ++k) {
                        const T a = alpha * A(i,k);
                        for (std::size_t j = j0; j < je; ++j)
                            C(i,j) += a * B(k,j);
                    }
            }
        }
    }
}
```

The tiling strategy ensures that the working set of each inner loop fits in L1/L2 cache, dramatically reducing cache misses for large matrices.

### Outer Product and Rank-k Updates

```cpp
// Outer product: C = u * v^T
template <typename T>
Matrix<T> outer(const std::vector<T>& u, const std::vector<T>& v) {
    Matrix<T> C(u.size(), v.size(), T{0});
    for (std::size_t i = 0; i < u.size(); ++i)
        for (std::size_t j = 0; j < v.size(); ++j)
            C(i, j) = u[i] * v[j];
    return C;
}

// Rank-1 update: A = A + alpha * u * v^T  (in-place)
template <typename T>
void rank1_update(Matrix<T>& A, const std::vector<T>& u,
                  const std::vector<T>& v, T alpha) {
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            A(i, j) += alpha * u[i] * v[j];
}
```

### Sparse Matrix Operations (CSR Format)

The library provides a `SparseCSR<T>` type for Compressed Sparse Row representation, with efficient SpMM and SpMV operations:

```cpp
template <typename T>
class SparseCSR {
public:
    static SparseCSR from_dense(const Matrix<T>& m);
    Matrix<T> to_dense() const;
    double sparsity() const noexcept;
    std::pair<size_type, size_type> row_range(size_type i) const;
    // ...
};

// Sparse Matrix × Dense Matrix: C = alpha*A*B + beta*C
template <typename T>
void spmm(const SparseCSR<T>& A, const Matrix<T>& B, Matrix<T>& C,
          T alpha, T beta) {
    for (std::size_t i = 0; i < A.rows(); ++i) {
        auto [rs, re] = A.row_range(i);
        for (std::size_t ptr = rs; ptr < re; ++ptr) {
            const T val = alpha * A.values()[ptr];
            const std::size_t k = A.col_idx()[ptr];
            for (std::size_t j = 0; j < B.cols(); ++j)
                C(i,j) += val * B(k,j);
        }
    }
}
```

### Multi-Vector Operations

Element-wise matrix operations including scale, add, Hadamard product, FMA, and per-row/per-column scaling:

```cpp
template <typename T>
Matrix<T> hadamard(const Matrix<T>& A, const Matrix<T>& B) {
    Matrix<T> C(A.rows(), A.cols(), T{0});
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            C(i, j) = A(i, j) * B(i, j);
    return C;
}

template <typename T>
void row_scale(Matrix<T>& A, const std::vector<T>& factors) {
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            A(i, j) *= factors[i];
}
```

---

## 7. Module 4 — Memory Tagging Extension (MTE) Simulation

### Overview

ARM's Memory Tagging Extension (MTE) provides hardware-assisted memory safety by associating 4-bit tags with memory allocations. This module provides MTE support on ARMv8.5-A hardware, using the `irg`, `stg`, and `ldg` instructions for tag generation, storage, and loading.

### API

```cpp
namespace navexa::mte {
    void    init_protection();                    // Enable MTE via prctl
    void*   malloc(size_t size);                  // Tagged memory allocation
    void    free(void* ptr, size_t size);          // Tag-clearing deallocation
    uint8_t get_pointer_tag(const void* ptr);      // Extract tag from pointer
    uint8_t get_memory_tag(const void* ptr);        // Read hardware memory tag
    bool    is_valid_pointer(const void* ptr);      // Tag match validation
}
```

### MTE Initialization

MTE protection is enabled per-thread using the Linux `prctl` system call:

```cpp
void init_protection() {
    if (prctl(PR_SET_TAGGED_ADDR_CTRL,
              PR_TAGGED_ADDR_ENABLE | PR_MTE_TCF_SYNC, 0, 0, 0) < 0) {
        std::perror("[MTE] prctl failed — QEMU/kernel may not support MTE");
    }
}
```

The `PR_MTE_TCF_SYNC` flag enables synchronous tag checking, which traps tag mismatches immediately as `SIGSEGV` rather than deferring them.

### Tagged Memory Allocation

```cpp
void* malloc(size_t size) {
    size_t aligned_size = (size + 15) & ~15;  // 16-byte granule alignment

    // Map memory with PROT_MTE flag
    void* ptr = mmap(nullptr, aligned_size,
                     PROT_READ | PROT_WRITE | PROT_MTE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) return nullptr;

    uint64_t tagged_ptr;
    asm("irg %0, %1" : "=r"(tagged_ptr) : "r"(ptr));  // Generate random tag

    // Apply tag to every 16-byte granule
    uint64_t current = tagged_ptr;
    for (size_t i = 0; i < aligned_size; i += 16) {
        asm volatile("stg %0, [%0]" : : "r"(current) : "memory");
        current += 16;
    }
    return reinterpret_cast<void*>(tagged_ptr);
}
```

The `irg` (Insert Random Tag) instruction generates a random 4-bit tag and embeds it in bits 59:56 of the pointer. The `stg` (Store Allocation Tag) instruction writes the tag to the physical memory's tag storage.

### Bounds Checking via Tag Comparison

```cpp
uint8_t get_pointer_tag(const void* ptr) {
    return static_cast<uint8_t>((reinterpret_cast<uint64_t>(ptr) >> 56) & 0xF);
}

uint8_t get_memory_tag(const void* ptr) {
    uint64_t mem_tag;
    asm volatile("ldg %0, [%1]" : "=r"(mem_tag) : "r"(ptr) : "memory");
    return static_cast<uint8_t>((mem_tag >> 56) & 0xF);
}

bool is_valid_pointer(const void* ptr) {
    return get_pointer_tag(ptr) == get_memory_tag(ptr);
}
```

This provides software-level bounds checking without triggering fatal hardware traps, allowing applications to detect use-after-free and buffer overflow conditions gracefully.

---

## 8. Module 5 — LSE Atomic Operations

### Overview

This module provides ARMv8.1 Large System Extensions (LSE) atomic primitives for building lock-free concurrent data structures. It includes:

- **CAS (Compare-And-Swap)** with acquire-release and relaxed semantics
- **LDADD (Atomic Add)** for fetch-and-add operations
- **Typed `Atomic64` wrapper** with load/store/CAS/fetch-add
- **`RefCount`** for atomic reference counting
- **`FreeList`** — a lock-free Treiber stack with ABA protection via 128-bit CASPAL

### Low-Level LSE Wrappers

```cpp
[[nodiscard]] inline uint64_t
cas64_acq_rel(uint64_t* ptr, uint64_t expected, uint64_t desired) noexcept {
    __asm__ volatile(
        "casal %[old], %[new_val], [%[addr]]"
        : [old] "+r"(expected)
        : [new_val] "r"(desired), [addr] "r"(ptr)
        : "memory"
    );
    return expected;  // returns actual value found in memory
}

[[nodiscard]] inline uint64_t
ldadd64_acq_rel(uint64_t* ptr, uint64_t addend) noexcept {
    uint64_t result;
    __asm__ volatile(
        "ldaddal %[add], %[res], [%[addr]]"
        : [res] "=r"(result)
        : [add] "r"(addend), [addr] "r"(ptr)
        : "memory"
    );
    return result;  // returns old value
}
```

### Typed Atomic64 Wrapper

```cpp
struct Atomic64 {
    explicit Atomic64(uint64_t initial = 0) noexcept : val_(initial) {}

    [[nodiscard]] uint64_t load() const noexcept {
        uint64_t v;
        __asm__ volatile("ldar %[v], [%[addr]]"
            : [v] "=r"(v) : [addr] "r"(&val_) : "memory");
        return v;
    }

    void store(uint64_t desired) noexcept {
        __asm__ volatile("stlr %[v], [%[addr]]"
            : : [v] "r"(desired), [addr] "r"(&val_) : "memory");
    }

    [[nodiscard]] bool compare_exchange(uint64_t& expected,
                                         uint64_t desired) noexcept {
        const uint64_t prev = cas64_acq_rel(&val_, expected, desired);
        if (prev == expected) return true;
        expected = prev;
        return false;
    }

    [[nodiscard]] uint64_t fetch_add(uint64_t addend) noexcept {
        return ldadd64_acq_rel(&val_, addend);
    }
};
```

### Lock-Free FreeList with CASPAL (128-bit CAS)

The FreeList uses a Treiber stack with ABA protection. The head is a 16-byte-aligned struct containing both a pointer and a monotonically increasing stamp:

```cpp
struct alignas(16) HeadSlot {
    uint64_t ptr;    // current head pointer
    uint64_t stamp;  // ABA stamp (incremented on pop)
};

// 128-bit CAS using ARMv8.1 CASPAL instruction
static bool caspal128(uint64_t* addr,
                      uint64_t exp_lo, uint64_t exp_hi,
                      uint64_t new_lo, uint64_t new_hi) noexcept {
    register uint64_t r0 __asm__("x0") = exp_lo;
    register uint64_t r1 __asm__("x1") = exp_hi;
    register uint64_t r2 __asm__("x2") = new_lo;
    register uint64_t r3 __asm__("x3") = new_hi;
    __asm__ volatile(
        "caspal %[lo], %[hi], %[nlo], %[nhi], [%[ptr]]"
        : [lo] "+r"(r0), [hi] "+r"(r1)
        : [nlo] "r"(r2), [nhi] "r"(r3), [ptr] "r"(addr)
        : "memory"
    );
    return (r0 == exp_lo) && (r1 == exp_hi);
}
```

The `push` operation links a new node and swaps the head. The `pop` operation increments the stamp on each successful removal, making ABA collisions detectable. Both operations use `yield` on contention to reduce power consumption:

```cpp
void FreeList::push(FreeNode* node) noexcept {
    uint64_t* slot = reinterpret_cast<uint64_t*>(&head_data_);
    while (true) {
        const uint64_t cur_ptr   = head_data_.ptr;
        const uint64_t cur_stamp = head_data_.stamp;
        node->next.store(cur_ptr);
        if (caspal128(slot, cur_ptr, cur_stamp,
                      reinterpret_cast<uint64_t>(node), cur_stamp))
            return;
        __asm__ volatile("yield" ::: "memory");  // power-saving hint
    }
}
```

### Atomic Reference Counter

```cpp
class RefCount {
public:
    explicit RefCount(uint32_t initial = 1) noexcept : count_(initial) {}

    void retain() noexcept {
        uint32_t old_val, inc = 1u;
        __asm__ volatile(
            "ldaddal %w[inc], %w[old], [%[addr]]"
            : [old] "=r"(old_val)
            : [inc] "r"(inc), [addr] "r"(&count_)
            : "memory");
    }

    [[nodiscard]] bool release() noexcept {
        uint32_t old;
        __asm__ volatile(
            "ldaddal %w[neg], %w[old], [%[addr]]"
            : [old] "=r"(old)
            : [neg] "r"(static_cast<uint32_t>(-1)), [addr] "r"(&count_)
            : "memory");
        if (old == 1u) {
            __asm__ volatile("dmb ish" ::: "memory");
            return true;  // last reference — caller should free
        }
        return false;
    }
};
```

---

## 9. Module 6 — Virtualization & Hypervisor Helpers

### Overview

This module provides software abstractions for ARMv8-A virtualization features, modeling VM context switching, stage-2 MMU page table entries, and virtual interrupt handling. These are pure C++ abstractions intended for hypervisor prototypes, simulators, and OS research.

### VM Context Snapshot

```cpp
struct VmContext {
    std::array<uint64_t, 31> gpr{};   // x0..x30
    uint64_t sp     = 0;              // stack pointer (EL1)
    uint64_t pc     = 0;              // program counter
    uint64_t pstate = 0;              // PSTATE register
    uint32_t vmid   = 0;              // VM ID (VTTBR_EL2.VMID)

    void reset() { gpr.fill(0); sp = pc = pstate = 0; }
    bool is_valid() const { return vmid != 0; }

    uint64_t& x(std::size_t n) {
        if (n > 30) throw std::out_of_range("x0..x30 only");
        return gpr[n];
    }
};
```

### Stage-2 MMU Page Table Entry

Models the intermediate physical address (IPA) to physical address (PA) translation with memory attributes and permissions:

```cpp
enum class Stage2MemAttr : uint8_t {
    Device_nGnRnE = 0x00,   // strongly ordered device
    Normal_NC     = 0x05,   // non-cacheable
    Normal_WB     = 0x0F,   // write-back cacheable (most RAM)
};

enum class Stage2Perm : uint8_t {
    None = 0b00, Read = 0b01, Write = 0b10, RW = 0b11,
};

struct Stage2Entry {
    uint64_t      ipa, pa;
    Stage2MemAttr attr  = Stage2MemAttr::Normal_WB;
    Stage2Perm    perm  = Stage2Perm::RW;
    bool          valid = false;
    bool          af    = false;  // access flag

    uint64_t translate(uint64_t input_ipa) const {
        if (!valid) throw std::runtime_error("entry not valid");
        if (input_ipa != ipa) throw std::runtime_error("IPA mismatch");
        return pa;
    }
    bool permits_read()  const { return valid && (static_cast<uint8_t>(perm) & 0b01); }
    bool permits_write() const { return valid && (static_cast<uint8_t>(perm) & 0b10); }
};
```

### Virtual Interrupt Handling

```cpp
struct VirtInterrupt {
    uint32_t    intid    = 0;
    VirtIntType type     = VirtIntType::IRQ;
    uint8_t     priority = 0;       // 0 = highest priority
    bool        pending  = false;
    bool        enabled  = false;
    uint32_t    vmid     = 0;

    void assert_int()  { if (enabled) pending = true; }
    void deassert_int() { pending = false; }
    bool should_deliver() const { return enabled && pending; }
};
```

### Context Switch Simulation

```cpp
inline ContextSwitchResult simulate_context_switch(VmContext& from,
                                                    VmContext& to) {
    if (!from.is_valid() || !to.is_valid())
        return { false, from.vmid, to.vmid, 0 };

    ContextSwitchResult result;
    result.success    = true;
    result.from_vmid  = from.vmid;
    result.to_vmid    = to.vmid;
    result.cycles_est = 200;  // typical ARMv8 VM switch ~200 cycles
    return result;
}
```

---

## 10. Testing Strategy & Integration Tests

### Unit Testing

Each module has dedicated unit tests using Google Test. Tests are cross-compiled for AArch64 and executed via QEMU:

```cmake
# tests/CMakeLists.txt
add_executable(test_sve tests/unit/test_sve.cpp)
target_link_libraries(test_sve armv8lib GTest::gtest_main)
add_test(NAME test_sve
  COMMAND qemu-aarch64 -L /usr/aarch64-linux-gnu $<TARGET_FILE:test_sve>)
```

**Test coverage includes:**

| Module | Tests | Key Scenarios |
|--------|-------|---------------|
| SVE | 25+ tests | Predicate construction, predicated load/store, merging arithmetic, gather/scatter, vecop loop, f64 ops |
| MathExt | 20+ tests | exp/log/sin/cos known values, inverse consistency, tail handling, activation functions, fixed-point round-trip |
| SME | 6 tests | 2×2 and non-square GEMM, alpha/beta scaling, dimension mismatch errors, double precision |
| Atomics | 8 tests | Load/store, CAS success/failure, fetch-add/sub, RefCount lifecycle, FreeList LIFO, multi-threaded stress |
| MTE | Manual test | Tag generation, bounds checking, intentional tag mismatch detection |

Example: SVE predicate unit test:

```cpp
TEST(SvePred, WhileLt_TailOf1_OnlyLane0Active) {
    SvePred pg = pred_while_lt(4, 5);
    EXPECT_TRUE(pg.active(0));
    EXPECT_FALSE(pg.active(1));
    EXPECT_FALSE(pg.active(2));
    EXPECT_FALSE(pg.active(3));
}
```

Example: MathExt exp/log inverse test:

```cpp
TEST(MathLog, LogIsInverseOfExp) {
    float vals[4] = {0.5f, 1.0f, 2.0f, 7.389f};
    float exp_out[4] = {}, log_out[4] = {};
    vec_exp_f32(exp_out, vals, 4);
    vec_log_f32(log_out, exp_out, 4);
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(log_out[i], vals[i], 1e-3f);
}
```

Example: Multi-threaded atomics stress test:

```cpp
TEST(FreeList, MultiThreadedStress) {
    FreeList fl;
    std::atomic<int> total_popped{0};
    // 4 producer threads × 1000 push operations
    // 4 consumer threads × 1000 pop operations
    // Verify: all nodes produced and consumed exactly once
    EXPECT_TRUE(fl.empty());
    EXPECT_EQ(total_popped.load(), N_THREADS * M_OPS);
}
```

### Integration Tests

The integration test suite (`tests/integration/test_combined.cpp`) exercises cross-module data flows:

1. **SVE → MathExt pipeline:** SVE vecop add → exp → dot product
2. **Non-aligned sin/cos pipeline:** SVE add (length 7) → sincos → Pythagorean identity verification
3. **SME → MathExt activation:** GEMM layer → sigmoid activation (ML inference pattern)
4. **SME GEMM → outer product → sparse round-trip**
5. **Full ML inference simulation:** GEMM → bias add → activation → softmax
6. **Virtualization + computation:** Context-switch with preserved computation state
7. **Fixed-point pipeline:** float → Q1.15 → matrix ops → back to float
8. **SVE gather/scatter + math pipeline**

Example integration test — ML inference pipeline:

```cpp
TEST(IntegrationSveMath, VecopThenExpThenDot) {
    float a[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float b[8] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float sum[8] = {};
    navexa::sve::vecop_f32(sum, a, b, 8, navexa::sve::SveOp::ADD);

    float exp_out[8] = {};
    navexa::math::vec_exp_f32(exp_out, sum, 8);

    float ones[8] = {1,1,1,1,1,1,1,1};
    float dot = navexa::math::vec_dot_f32(exp_out, ones, 8);

    float ref = 0.0f;
    for (int i = 0; i < 8; ++i) ref += expf(a[i] + 0.1f);
    expect_near_rel(dot, ref, "SVE→exp→dot pipeline");
}
```

---

## 11. Performance Benchmarking

The library includes a standalone micro-benchmark suite (`benchmarks/bench_sve.cpp`) that measures throughput in millions of elements per second for all public functions.

### Benchmark Design

```cpp
static void run_bench(const std::string& name, size_t N,
                      std::function<void()> fn) {
    for (int w = 0; w < kWarmupIter; ++w) fn();  // warm-up
    long long iters = 0;
    double start_ms = now_ms(), elapsed = 0.0;
    do {
        fn();
        ++iters;
        elapsed = now_ms() - start_ms;
    } while (elapsed < kMinTimeMs);    // run for at least 200ms
    double mops = (double(N) * double(iters)) / (elapsed * 1e3);
    g_results.push_back({name, mops, elapsed, iters});
}
```

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `kMinTimeMs` | 200 ms | Minimum measurement window per benchmark |
| `kMedN` | 4096 | Array length fitting L1/L2 cache |
| `kLargeN` | 65536 | L2/L3 boundary stress test |
| `kWarmupIter` | 3 | Warm-up iterations before timing |

The benchmark uses 64-byte aligned buffers and volatile sinks to prevent dead-code elimination:

```cpp
struct AlignedBuf {
    std::vector<float> storage;
    float* ptr;
    explicit AlignedBuf(size_t n, float fill = 0.0f) {
        storage.resize(n + 16, fill);
        uintptr_t raw = reinterpret_cast<uintptr_t>(storage.data());
        uintptr_t aligned = (raw + 63u) & ~static_cast<uintptr_t>(63u);
        ptr = reinterpret_cast<float*>(aligned);
    }
};
```

### Build and Run

```bash
aarch64-linux-gnu-g++ -O3 -march=armv8.5-a+memtag+lse \
    -I include \
    benchmarks/bench_sve.cpp \
    src/sve/sve_emulation.cpp \
    src/sve/neon_fallback.cpp \
    src/mathext/transcendentals.cpp \
    -o bench_sve
qemu-aarch64 -L /usr/aarch64-linux-gnu ./bench_sve
```

---

## 12. Build System & CI/CD Pipeline

### CMake Configuration

The root `CMakeLists.txt` configures the library with C++20, ARMv8.5-A architecture flags, and enables testing:

```cmake
cmake_minimum_required(VERSION 3.20)
project(armv8lib VERSION 0.1.0 LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Architecture flags — +lse for atomics, +memtag for MTE
add_compile_options(
    -march=armv8.5-a+memtag+lse
    -Wall -Wextra
)

file(GLOB_RECURSE LIB_SOURCES src/**/*.cpp)
add_library(armv8lib STATIC ${LIB_SOURCES})
target_include_directories(armv8lib PUBLIC include)
target_include_directories(armv8lib PRIVATE src)

enable_testing()
add_subdirectory(tests)
```

### Architecture Flags

| Flag | Extension | Required By |
|------|-----------|-------------|
| `-march=armv8.5-a` | Base ARMv8.5-A | Core library |
| `+memtag` | Memory Tagging Extension | MTE module (`irg`, `stg`, `ldg`) |
| `+lse` | Large System Extensions | Atomics module (`casal`, `ldaddal`, `caspal`) |

### Code Style

The `.clang-format` file enforces consistent code style across all contributions:

```yaml
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
PointerAlignment: Left
SortIncludes: true
```

Format check before commits:
```bash
clang-format -i src/**/*.cpp include/**/*.h
```

---

## 13. Use Cases & Applications

### AI/ML Inference

The library provides a complete pipeline for neural network inference on ARM:

```
Input Data → GEMM (weight multiply) → Bias Add (vec_add_f32)
           → Activation (sigmoid/tanh/relu) → Softmax (vec_exp_f32 + reduce)
```

Key functions: `gemm()`, `vec_add_f32()`, `vec_sigmoid_f32()`, `vec_tanh_f32()`, `vec_relu_f32()`, `vec_exp_f32()`, `vec_dot_f32()`

### High-Performance Computing (HPC)

- **Vectorized transcendentals** (`vec_sin_f32`, `vec_cos_f32`, `vec_exp_f32`) for physics simulations
- **SVE-style predicated loops** for natural tail handling without scalar cleanup
- **Sparse matrix operations** (`spmm`, `spmv`) for finite element methods

### Embedded Systems & DSP

- **Fixed-point conversion** (`f32_to_q15`, `q15_to_f32`) for interfacing with DSP peripherals
- **NEON-optimized kernels** for real-time signal processing
- **Compact static library** output for resource-constrained targets

### Memory Safety & Security

- **MTE-based bounds checking** for detecting buffer overflows and use-after-free
- **Software tag validation** for graceful error handling without crashes

### Operating System & Hypervisor Development

- **VM context switch simulation** for prototyping hypervisors
- **Stage-2 MMU abstractions** for memory virtualization research
- **Virtual interrupt handling** for GIC (Generic Interrupt Controller) prototyping

### Concurrent Data Structures

- **Lock-free free-list** with ABA protection for memory allocators
- **Atomic reference counting** for shared ownership in multi-threaded systems
- **LSE atomic primitives** (CAS, LDADD) for custom synchronization

---

## 14. Conclusion

The navexa ARMv8-A Library delivers a comprehensive, production-quality implementation of advanced ARM architectural features on the base ARMv8-A platform. By combining NEON intrinsics, inline assembly, and portable C++ algorithms, the library bridges the gap between the widely deployed ARMv8-A specification and the capabilities available only in newer Armv9 processors.

**Key achievements:**

- **6 fully implemented modules** covering SIMD, matrix operations, memory safety, concurrency, virtualization, and vectorized mathematics
- **Zero ARM hardware requirement** — the entire development, testing, and benchmarking workflow runs on x86 via QEMU user-mode emulation
- **Cross-module integration** — 8 integration test scenarios verify end-to-end data flow across modules
- **Production practices** — CI/CD pipeline, code formatting enforcement, structured testing with Google Test, and comprehensive documentation
- **Practical applicability** — direct use cases in AI/ML inference, HPC, embedded systems, operating systems, and concurrent programming

The library serves both as a practical toolbox for ARM software development and as an educational resource demonstrating ARM architecture concepts from NEON SIMD to MTE memory tagging to LSE lock-free concurrency.

---

**Repository:** [github.com/adarsh-v-h/navexa_ARMv8-A_Library](https://github.com/adarsh-v-h/navexa_ARMv8-A_Library)  
**Documentation:** [deepwiki.com/adarsh-v-h/navexa_ARMv8-A_Library](https://deepwiki.com/adarsh-v-h/navexa_ARMv8-A_Library/1-project-overview)

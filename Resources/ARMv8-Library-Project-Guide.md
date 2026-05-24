# ARMv8-A Library — Zero to One: Team Project Guide

A complete guide for a 4-person team building a production C/C++ ARMv8-A library, from first-day setup to a clean, testable, mergeable codebase — with no ARM hardware in hand.

---

## Part 1 — No ARM Hardware? Here's How You Work Anyway

This is the most critical problem to solve first, because everything else — writing code, testing, CI — depends on it.

### The Solution Stack

You will use **two things together**:

**Cross-Compiler**: A GCC toolchain that runs on your x86 Linux machine but produces binaries for AArch64 (ARMv8-A 64-bit). It does not execute the binary — it only compiles it.

**QEMU User-Mode Emulator**: A program that lets your x86 machine *run* AArch64 binaries transparently, as if they were native. `qemu-aarch64` intercepts AArch64 syscalls and translates them. This is how projects like sse2neon run their entire ARM test suite on x86 CI machines.

Together: you write and compile on your machine → QEMU runs the binary → you see output and test results. No ARM board needed.

### Step-by-Step: Environment Setup (Every Team Member Does This)

All members must be on **Ubuntu 22.04 or 24.04** (or Pop!_OS, which you already use). If someone is on Windows, they must use WSL2 with Ubuntu.

#### Step 1 — Install the cross-compiler and QEMU

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
  valgrind
```

Verify the toolchain:
```bash
aarch64-linux-gnu-g++ --version
# Should print: aarch64-linux-gnu-g++ 11.x or 12.x
```

#### Step 2 — Verify QEMU can run ARM binaries

```bash
# Write a quick test
cat > /tmp/test.cpp << 'EOF'
#include <stdio.h>
int main() { printf("Running on AArch64 via QEMU\n"); return 0; }
EOF

# Cross-compile it
aarch64-linux-gnu-g++ /tmp/test.cpp -o /tmp/test_arm

# Run it via QEMU — this should just work
qemu-aarch64 -L /usr/aarch64-linux-gnu /tmp/test_arm
```

You should see: `Running on AArch64 via QEMU`

#### Step 3 — Verify intrinsics compile

Some of your library will use ARM NEON intrinsics (`#include <arm_neon.h>`). Verify those compile:

```bash
cat > /tmp/neon_test.cpp << 'EOF'
#include <arm_neon.h>
#include <stdio.h>
int main() {
    float32x4_t v = vdupq_n_f32(3.14f);
    float result[4];
    vst1q_f32(result, v);
    printf("NEON vector: %f %f %f %f\n", result[0], result[1], result[2], result[3]);
    return 0;
}
EOF

aarch64-linux-gnu-g++ /tmp/neon_test.cpp -o /tmp/neon_test -march=armv8-a
qemu-aarch64 -L /usr/aarch64-linux-gnu /tmp/neon_test
```

You should see: `NEON vector: 3.140000 3.140000 3.140000 3.140000`

#### Step 4 — Set QEMU_LD_PREFIX (convenience shortcut)

Every time you run a cross-compiled binary, you need `-L /usr/aarch64-linux-gnu` to tell QEMU where the ARM shared libraries live. Set it once in your shell:

```bash
echo 'export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu' >> ~/.bashrc
source ~/.bashrc
```

Now you can just run `./your_arm_binary` directly without the flag.

### Compiler Flags Reference

Use these when compiling any part of the library:

| Flag | Purpose |
|---|---|
| `--target=aarch64-linux-gnu` | Tell compiler: AArch64 target |
| `-march=armv8-a` | Base ARMv8-A instruction set |
| `-march=armv8-a+crypto` | Include AES/SHA crypto extensions |
| `-march=armv8-a+sve` | Include SVE (GCC 9+ only) |
| `-march=armv8.5-a+rng` | Include RNDR random number instruction |
| `-O2` | Standard optimization level |
| `-g` | Debug symbols (include during dev) |

Use feature flags per module — don't set SVE globally if half the library doesn't need it.

---

## Part 2 — Dividing the 10 Functionalities Across 4 People

### Assignment

With 10 functionalities and 4 people, split as follows (suggested, adjust by interest):

| Person | Functionalities |
|---|---|
| Person A | 1 — SVE/NEON, 9 — Math/Transcendentals |
| Person B | 2 — SME Matrix, 8 — Virtualization Helpers |
| Person C | 3 — MTE Simulation, 5 — Atomics |
| Person D | 4 — Crypto Extensions, 6 — RNG, 7 — Power/Perf, 10 — Security/PAC |

Person D gets four items because 7 and 10 are smaller in scope — they're mostly wrappers and constants, not full algorithm implementations.

### How to Work Without Breaking Each Other

The answer is **strict module boundaries via directory structure + a clean public API layer**. As long as no two people own the same files, merges will rarely conflict.

The rule is simple:
- Each functionality lives in its own directory under `src/`
- Each person only modifies files inside their assigned directories
- All cross-module calls go through public headers in `include/armv8lib/`
- The `include/` headers are co-owned — changes there require a PR review by at least one other person

This means Person A working on SVE can never accidentally break Person C's MTE work, because they literally touch different files.

---

## Part 3 — GitHub Setup: Everything You Need

### Repository Structure

```
armv8-lib/
├── .github/
│   ├── workflows/
│   │   └── ci.yml              # GitHub Actions: build + test on every PR
│   └── PULL_REQUEST_TEMPLATE.md
├── cmake/
│   └── toolchain-aarch64.cmake # Cross-compilation toolchain file
├── include/
│   └── armv8lib/               # All public headers — the API surface
│       ├── sve.h
│       ├── sme.h
│       ├── mte.h
│       ├── crypto.h
│       ├── atomics.h
│       ├── rng.h
│       ├── power.h
│       ├── virt.h
│       ├── mathext.h
│       └── security.h
├── src/
│   ├── sve/                    # Person A
│   │   ├── sve_emulation.cpp
│   │   └── neon_fallback.cpp
│   ├── sme/                    # Person B
│   │   ├── gemm.cpp
│   │   └── outer_product.cpp
│   ├── mte/                    # Person C
│   │   ├── mte_sim.cpp
│   │   └── bounds_check.cpp
│   ├── crypto/                 # Person D
│   │   ├── aes.cpp
│   │   └── sha.cpp
│   ├── atomics/                # Person C
│   │   └── atomic_ops.cpp
│   ├── rng/                    # Person D
│   │   └── rndr.cpp
│   ├── power/                  # Person D
│   │   └── power_mgmt.cpp
│   ├── virt/                   # Person B
│   │   └── vhe_helpers.cpp
│   ├── mathext/                # Person A
│   │   └── transcendentals.cpp
│   └── security/               # Person D
│       └── pac_bti.cpp
├── tests/
│   ├── unit/
│   │   ├── test_sve.cpp
│   │   ├── test_sme.cpp
│   │   ├── test_mte.cpp
│   │   ├── test_crypto.cpp
│   │   ├── test_atomics.cpp
│   │   ├── test_rng.cpp
│   │   ├── test_power.cpp
│   │   ├── test_virt.cpp
│   │   ├── test_mathext.cpp
│   │   └── test_security.cpp
│   └── integration/
│       └── test_combined.cpp
├── benchmarks/
│   └── bench_sve.cpp           # Optional: add perf benchmarks later
├── docs/
│   └── design.md               # High-level architecture decisions
├── CMakeLists.txt              # Root CMake file
├── .clang-format               # Enforced code style
├── .gitignore
└── README.md
```

### CMake Setup

`CMakeLists.txt` (root):
```cmake
cmake_minimum_required(VERSION 3.20)
project(armv8lib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all source files
file(GLOB_RECURSE LIB_SOURCES src/**/*.cpp)

# Build as a static library
add_library(armv8lib STATIC ${LIB_SOURCES})
target_include_directories(armv8lib PUBLIC include)

# Tests
enable_testing()
add_subdirectory(tests)
```

`cmake/toolchain-aarch64.cmake`:
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

### Build Commands (Everyone Uses the Same)

```bash
# From repo root:
mkdir build && cd build

# Configure with cross-compilation toolchain
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-aarch64.cmake \
         -DCMAKE_BUILD_TYPE=Debug

# Build everything
cmake --build . -j$(nproc)

# Run tests through CTest (which uses QEMU automatically)
ctest --output-on-failure
```

### .clang-format (paste this at root, everyone uses it)

```yaml
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
PointerAlignment: Left
SortIncludes: true
```

Run before every commit: `clang-format -i src/**/*.cpp include/**/*.h`

### .gitignore

```
build/
*.o
*.a
*.so
*.out
.vscode/
.cache/
compile_commands.json
```

---

## Part 4 — Branch Strategy

Use **GitHub Flow** — it's simple, works well for small teams, and keeps `main` always stable.

### The Model

```
main  ──────────────────────────────────────────────────> (always stable)
         |           |              |
   feat/sve-emul  feat/mte-sim  feat/crypto-aes
         |           |              |
        PR #1       PR #2          PR #3
```

### Branch Naming Convention

```
feat/<module>-<short-description>
fix/<module>-<bug-name>
docs/<what>
test/<module>-<what>
```

Examples:
```
feat/sve-neon-fallback
feat/mte-bounds-checking
fix/crypto-aes-endianness
test/rng-statistical-tests
```

### Rules for `main`

Enable these in GitHub → Settings → Branches → Add branch protection rule:
- Require pull request before merging: **ON**
- Require at least 1 approving review: **ON**
- Require status checks to pass (CI): **ON**
- Do not allow force pushes: **ON**

Nobody pushes directly to `main` — ever.

### Day-to-Day Workflow for Each Person

```bash
# Start new work
git checkout main
git pull origin main
git checkout -b feat/mte-bounds-checking

# Write code, make commits
git add src/mte/bounds_check.cpp tests/unit/test_mte.cpp
git commit -m "feat(mte): add basic tag simulation with 4-bit granule"

# Push and open PR
git push origin feat/mte-bounds-checking
# Then open PR on GitHub → request review from 1 teammate

# Keep branch up to date with main (avoid big merges later)
git fetch origin
git rebase origin/main
```

### Commit Message Convention

Use this format consistently — it makes the git log readable and makes it easy to see what changed where:

```
type(module): short description

Examples:
feat(sve): add 128-bit vector add with NEON fallback
fix(crypto): correct AES key schedule endianness
test(mte): add use-after-free detection test
docs(rng): document RNDR fallback behavior
refactor(atomics): reduce branching in cmpxchg path
```

---

## Part 5 — Testing Strategy

### Framework: Google Test (gtest)

It's the standard for C++ unit testing. Integrates with CMake out of the box.

Install it:
```bash
sudo apt install -y libgtest-dev
```

Or use CMake's FetchContent (preferred — everyone gets the same version automatically):

Add to `tests/CMakeLists.txt`:
```cmake
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Add each test executable
foreach(module sve sme mte crypto atomics rng power virt mathext security)
  add_executable(test_${module} unit/test_${module}.cpp)
  target_link_libraries(test_${module} armv8lib GTest::gtest_main)
  add_test(NAME ${module} COMMAND test_${module})
endforeach()
```

### How to Write a Unit Test (Example: RNG)

`tests/unit/test_rng.cpp`:
```cpp
#include <gtest/gtest.h>
#include "armv8lib/rng.h"

// Test 1: Basic generation — should not return 0 repeatedly
TEST(RNG, NonZeroOutput) {
    uint64_t val = armv8lib_rng_generate();
    // We can't assert exact value, but 0 from a real RNG is astronomically rare
    EXPECT_NE(val, 0);
}

// Test 2: Two calls should produce different values
TEST(RNG, UniqueValues) {
    uint64_t a = armv8lib_rng_generate();
    uint64_t b = armv8lib_rng_generate();
    EXPECT_NE(a, b);
}

// Test 3: Seeded PRNG fallback — deterministic given same seed
TEST(RNG, DeterministicFallback) {
    armv8lib_rng_seed(42);
    uint64_t first  = armv8lib_rng_fallback();
    armv8lib_rng_seed(42);
    uint64_t second = armv8lib_rng_fallback();
    EXPECT_EQ(first, second);
}
```

Run all tests:
```bash
cd build
ctest --output-on-failure
```

Run a single module's tests:
```bash
./tests/test_rng
./tests/test_crypto
```

### Testing Layers

| Layer | What it tests | Who writes it |
|---|---|---|
| Unit tests | Single function, isolated behavior | The person who owns that module |
| Integration tests | Two modules interacting (e.g., crypto using RNG) | Whoever initiates the interaction |
| Build test | Does everything compile cleanly | CI does this automatically |

### Testing Intrinsics-Heavy Code

Some functions will call ARM intrinsics directly. These will only be testable via QEMU (or real hardware). For functions that must work on host x86 too (for reference/fallback implementations), guard them:

```cpp
#ifdef __aarch64__
    // ARM-only path using NEON intrinsics
    return vaddq_f32(a, b);
#else
    // Scalar fallback for host testing
    float32x4_t result;
    for (int i = 0; i < 4; i++) result[i] = a[i] + b[i];
    return result;
#endif
```

---

## Part 6 — GitHub Actions CI

Every PR should automatically build and run all tests. This catches broken code before review.

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-22.04

    steps:
    - uses: actions/checkout@v4

    - name: Install cross-compiler and QEMU
      run: |
        sudo apt-get update
        sudo apt-get install -y \
          gcc-aarch64-linux-gnu \
          g++-aarch64-linux-gnu \
          qemu-user-static \
          cmake ninja-build

    - name: Configure
      run: |
        mkdir build
        cmake -B build \
          -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64.cmake \
          -DCMAKE_BUILD_TYPE=Debug \
          -G Ninja

    - name: Build
      run: cmake --build build -j4

    - name: Test
      run: |
        export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu
        cd build && ctest --output-on-failure
```

When you open a PR, this runs automatically. The PR cannot be merged if CI is red.

---

## Part 7 — Things You Must Decide Before Writing a Single Line of Code

These are the things teams skip and regret. Do them in your first team meeting.

### 1. Lock the Public API Style

Decide upfront: does your library use a C API, a C++ class API, or a namespace-based API? All 10 modules must follow the same pattern.

Recommended for a C++ library: **namespace + free functions**

```cpp
namespace armv8lib {
namespace sve {
    void vector_add_f32(float* out, const float* a, const float* b, size_t len);
}
namespace crypto {
    void aes128_encrypt(uint8_t* out, const uint8_t* in, const uint8_t* key);
}
}
```

Write a `CONTRIBUTING.md` that states this convention so every person follows it.

### 2. Decide on Error Handling

What does a function return when hardware support is missing (e.g., no RNDR)?

Option A: Return a boolean success flag — `bool armv8lib_rng_generate(uint64_t* out)`  
Option B: Return a status enum — `armv8lib_status_t armv8lib_rng_generate(uint64_t* out)`  
Option C: Exceptions (C++ only)

Pick one and document it. Mixing styles across modules creates chaos.

### 3. Hardware Feature Detection at Runtime

Some features (crypto extensions, SVE, RNDR) may or may not be present on the target CPU. You need a detection mechanism so the library can fall back gracefully.

On Linux/AArch64, use `/proc/cpuinfo` or `getauxval(AT_HWCAP)`:

```cpp
#include <sys/auxv.h>
#include <asm/hwcap.h>

bool has_aes_hardware() {
    return (getauxval(AT_HWCAP) & HWCAP_AES) != 0;
}
bool has_sve() {
    return (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
}
```

Build a single `src/detect.cpp` with all detection functions early. Every module imports this.

### 4. Soft Link Between Modules (Dependency Rules)

Define which modules can call which. Without this, you'll end up with circular dependencies.

Suggested rule:
```
crypto    → can use rng, detect
sve       → can use detect only
mte       → no dependencies on other modules
atomics   → no dependencies
rng       → can use detect only
security  → can use detect only
mathext   → can use sve (for vectorized paths)
```

Draw this out on a whiteboard or a shared doc. If someone needs to break this rule, they must raise it as a PR discussion first.

### 5. Documentation Standard

Every public function in `include/armv8lib/` must have a Doxygen-style comment. Non-negotiable — the comment is part of the implementation.

```cpp
/**
 * @brief Generate a hardware random 64-bit value using RNDR instruction.
 *        Falls back to ChaCha-based PRNG if hardware RNG is unavailable.
 *
 * @param out  Pointer to store the generated value. Must not be NULL.
 * @return true if hardware RNG was used, false if fallback was used.
 */
bool armv8lib_rng_generate(uint64_t* out);
```

---

## Part 8 — Recommended First Week Plan

| Day | What happens |
|---|---|
| Day 1 | Everyone sets up environment (cross-compiler + QEMU). All run the NEON test to confirm it works. |
| Day 2 | Create the GitHub repo. Set up the directory structure. Add `.clang-format`, `.gitignore`, root `CMakeLists.txt`, and the toolchain file. CI workflow goes in. |
| Day 3 | Team meeting: lock the API style, error handling convention, module dependency rules. Write `CONTRIBUTING.md`. |
| Day 4 | Each person writes the header for their modules (`include/armv8lib/*.h`) — just the function signatures with doc comments. No implementation yet. Open a PR so the team reviews the API before code is written. |
| Day 5 | Implementations start. Each person also writes at least 2 unit tests for their first function. CI must be green before the week ends. |

After that, each person merges one feature PR per week minimum.

---

## Quick Reference Card

```
Build:       cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64.cmake && cmake --build build
Test all:    cd build && ctest --output-on-failure
Test one:    ./build/tests/test_crypto
Format:      clang-format -i src/**/*.cpp include/**/*.h
New branch:  git checkout main && git pull && git checkout -b feat/<module>-<desc>
```

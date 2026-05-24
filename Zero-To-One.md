# Zero to One — Your First Working ARM Function

The goal by the end of this: **one function that adds two NEON float vectors,
compiles for ARM, runs on your machine via QEMU, and passes a test.**
Every person on the team does this independently. When all four of you have it
working, you've proven the foundation — then you build on top.

---

## What You're Actually Building (Big Picture, Simple)

```
Your x86 machine
│
├── Cross-compiler (aarch64-linux-gnu-g++)
│     Takes your .cpp code → produces an ARM binary
│
├── QEMU (qemu-aarch64)
│     Takes that ARM binary → runs it on your x86 CPU (fakes ARM)
│
└── Google Test (gtest)
      A testing framework → your test.cpp calls your function,
      checks the output, prints PASS or FAIL
```

That's the whole stack. You write code → compile for ARM → QEMU runs it → test tells you if it works.

---

## Windows Users: Do This First

If you're on Windows, you cannot run this natively. You need WSL2 (Windows Subsystem for Linux). It gives you a real Ubuntu terminal inside Windows. Do this once and you're set.

**Step 1 — Open PowerShell as Administrator and run:**
```powershell
wsl --install
```
This installs WSL2 + Ubuntu automatically. Restart your machine when it asks.

**Step 2 — After restart, Ubuntu will open and ask you to create a username and password.**
Set anything you like. This is your Linux username — remember it.

**Step 3 — Verify it worked:**
```bash
uname -a
# Should print something like: Linux ... x86_64 GNU/Linux
```

From now on, every command in this guide runs inside the WSL2 Ubuntu terminal.

---

## Step 1 — Install the Tools (Every Person on Every Machine)

Open a terminal (Ubuntu/Pop!_OS native, or WSL2 on Windows) and run this exactly:

```bash
sudo apt update && sudo apt install -y \
  g++-aarch64-linux-gnu \
  qemu-user-static \
  cmake \
  git
```

What each thing does:
- `g++-aarch64-linux-gnu` — the cross-compiler. It's GCC but produces ARM binaries.
- `qemu-user-static` — the ARM emulator. Lets you *run* ARM binaries on x86.
- `cmake` — the build system. Tells the compiler what to compile and how.
- `git` — version control. You know this one.

**Verify it worked:**
```bash
aarch64-linux-gnu-g++ --version
```
You should see something like: `aarch64-linux-gnu-g++ 11.4.0`
If you see `command not found`, the install failed — re-run the apt command.

---

## Step 2 — One-Time QEMU Setup

When QEMU runs an ARM binary, it needs to find ARM system libraries (like the C standard library, but the ARM version). Tell it where to look:

```bash
echo 'export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu' >> ~/.bashrc
source ~/.bashrc
```

**What this does:** Sets an environment variable permanently. Without it, QEMU will say
`error while loading shared libraries` when you try to run anything.

Verify it stuck:
```bash
echo $QEMU_LD_PREFIX
# Should print: /usr/aarch64-linux-gnu
```

---

## Step 3 — Create the Project Folder

```bash
mkdir armv8-lib
cd armv8-lib
```

Now create this exact structure manually:

```bash
mkdir -p include/armv8lib
mkdir -p src/mathext
mkdir -p tests
```

Your folder now looks like:
```
armv8-lib/
├── include/
│   └── armv8lib/
├── src/
│   └── mathext/
└── tests/
```

---

## Step 4 — Write the Function (Header + Implementation)

You're going to implement one function: `vec_add_f32`.
It takes two float arrays, adds them element-by-element using ARM NEON (ARM's SIMD unit),
and writes the result into a third array.

### The header — what the function looks like from outside

Create `include/armv8lib/mathext.h`:

```cpp
#pragma once
#include <stddef.h>   // for size_t

namespace armv8lib {

/**
 * Add two float arrays element-wise using NEON vectorization.
 * out[i] = a[i] + b[i] for i in [0, len)
 * len must be a multiple of 4.
 */
void vec_add_f32(float* out, const float* a, const float* b, size_t len);

} // namespace armv8lib
```

### The implementation — the actual ARM code

Create `src/mathext/mathext.cpp`:

```cpp
#include "armv8lib/mathext.h"

// arm_neon.h is the ARM SIMD header — like <immintrin.h> is for x86 AVX.
// It gives us types like float32x4_t (a vector of 4 floats) and
// functions like vaddq_f32 (add two float32x4_t vectors).
#include <arm_neon.h>

namespace armv8lib {

void vec_add_f32(float* out, const float* a, const float* b, size_t len) {
    size_t i = 0;

    // Process 4 floats at a time using NEON
    for (; i + 4 <= len; i += 4) {
        // vld1q_f32: load 4 floats from memory into a NEON register
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

        // vaddq_f32: add the two 4-float vectors (one ARM instruction)
        float32x4_t vresult = vaddq_f32(va, vb);

        // vst1q_f32: store the result back to memory
        vst1q_f32(out + i, vresult);
    }

    // Scalar tail: handle remaining elements if len wasn't divisible by 4
    for (; i < len; i++) {
        out[i] = a[i] + b[i];
    }
}

} // namespace armv8lib
```

**What is `float32x4_t`?** It's an ARM type that represents 4 floats packed into a
single 128-bit NEON register. `vaddq_f32` adds all 4 pairs in one CPU instruction
instead of four separate additions. This is what SIMD means.

---

## Step 5 — Write the Test

Create `tests/test_mathext.cpp`:

```cpp
// Google Test header — gives us TEST(), EXPECT_FLOAT_EQ(), etc.
#include <gtest/gtest.h>
#include "armv8lib/mathext.h"

// TEST(SuiteName, TestName) defines one test.
// SuiteName groups related tests. TestName describes what this one checks.

TEST(VecAddF32, BasicAddition) {
    float a[4]   = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4]   = {10.0f, 20.0f, 30.0f, 40.0f};
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    armv8lib::vec_add_f32(out, a, b, 4);

    // EXPECT_FLOAT_EQ checks two floats are equal (handles floating point precision)
    EXPECT_FLOAT_EQ(out[0], 11.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 33.0f);
    EXPECT_FLOAT_EQ(out[3], 44.0f);
}

TEST(VecAddF32, ZeroVector) {
    float a[4]   = {5.0f, 6.0f, 7.0f, 8.0f};
    float b[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    armv8lib::vec_add_f32(out, a, b, 4);

    // Adding zero should return the original values unchanged
    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], 6.0f);
    EXPECT_FLOAT_EQ(out[2], 7.0f);
    EXPECT_FLOAT_EQ(out[3], 8.0f);
}
```

---

## Step 6 — Write the Build File (CMake)

CMake is the tool that knows: "here are the source files, here's the compiler,
here's how to link everything together." You write it once, then just run two commands to build.

### Toolchain file — tells CMake to use the ARM cross-compiler

Create `cmake/toolchain-aarch64.cmake`:

```cmake
# Tell CMake the target is Linux on AArch64 (64-bit ARM)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Use the ARM cross-compiler instead of the default x86 GCC
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Tell CMake where ARM libraries live (so it finds arm_neon.h etc.)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

### Root CMakeLists.txt — the main build instructions

Create `CMakeLists.txt` at the root of `armv8-lib/`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(armv8lib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Build the library ---
# Collect source files and build them as a static library (.a file)
add_library(armv8lib STATIC
    src/mathext/mathext.cpp
)
# Tell the compiler: look in include/ for headers like "armv8lib/mathext.h"
target_include_directories(armv8lib PUBLIC include)
# Compile flag: target ARMv8-A instruction set (enables arm_neon.h)
target_compile_options(armv8lib PRIVATE -march=armv8-a)


# --- Build and register the tests ---
enable_testing()

# Download Google Test automatically (no manual install needed)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Build the test executable
add_executable(test_mathext tests/test_mathext.cpp)
# Link it against: our library + gtest
target_link_libraries(test_mathext armv8lib GTest::gtest_main)
target_compile_options(test_mathext PRIVATE -march=armv8-a)

# Register it with CTest so `ctest` knows to run it
# IMPORTANT: the COMMAND line tells CTest to run it through QEMU
add_test(
  NAME mathext
  COMMAND qemu-aarch64 -L /usr/aarch64-linux-gnu $<TARGET_FILE:test_mathext>
)
```

**Why `qemu-aarch64 -L /usr/aarch64-linux-gnu` in the test command?**
Because `test_mathext` is an ARM binary. CTest (the test runner) can't run it
directly on your x86 machine. Prefixing it with `qemu-aarch64` tells CTest:
"run this binary through the ARM emulator."

---

## Step 7 — Your Final Folder Structure

Before building, confirm everything is in place:

```
armv8-lib/
├── cmake/
│   └── toolchain-aarch64.cmake
├── include/
│   └── armv8lib/
│       └── mathext.h
├── src/
│   └── mathext/
│       └── mathext.cpp
├── tests/
│   └── test_mathext.cpp
└── CMakeLists.txt
```

If anything is missing, create it now before moving on.

---

## Step 8 — Build It

```bash
# Make sure you're inside armv8-lib/
cd armv8-lib

# Create a build folder (keeps all compiled files separate from your source)
mkdir build
cd build

# Configure: tell CMake to use the ARM toolchain
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-aarch64.cmake

# Build: actually compile the library and test binary
cmake --build . -j$(nproc)
```

**What you'll see during cmake configure:**
```
-- The CXX compiler identification is GNU 11.4.0
-- Check for working CXX compiler: /usr/bin/aarch64-linux-gnu-g++ - works
-- Configuring done
```

**What you'll see during cmake build:**
```
[ 25%] Building CXX object CMakeFiles/armv8lib.dir/src/mathext/mathext.cpp.o
[ 50%] Linking CXX static library libarmv8lib.a
[ 75%] Building CXX object CMakeFiles/test_mathext.dir/tests/test_mathext.cpp.o
[100%] Linking CXX executable test_mathext
```

If it errors, the most common cause is a missing file. Read the error — it will tell you exactly which file it couldn't find.

**Verify the binary is actually ARM (not x86):**
```bash
file test_mathext
# Should print: test_mathext: ELF 64-bit LSB executable, ARM aarch64
```
If it says `x86-64`, the toolchain file wasn't picked up. Re-run cmake with the toolchain flag.

---

## Step 9 — Run the Test

```bash
# Still inside armv8-lib/build/
ctest --output-on-failure
```

**What you should see:**
```
Test project /home/you/armv8-lib/build
    Start 1: mathext
1/1 Test #1: mathext .........................   Passed    0.12 sec

100% tests passed, 0 tests failed out of 1
```

You can also run the test binary directly to see the full Google Test output:
```bash
qemu-aarch64 -L /usr/aarch64-linux-gnu ./test_mathext
```

Which prints:
```
[==========] Running 2 tests from 1 test suite.
[----------] 2 tests from VecAddF32
[ RUN      ] VecAddF32.BasicAddition
[       OK ] VecAddF32.BasicAddition (0 ms)
[ RUN      ] VecAddF32.ZeroVector
[       OK ] VecAddF32.ZeroVector (0 ms)
[----------] 2 tests from VecAddF32 (0 ms total)

[==========] 2 tests ran.
[  PASSED  ] 2 tests passed.
```

**You're at 1. An ARM function compiled, ran, and passed tests on an x86 machine.**

---

## What to Do When It Breaks (Common Errors)

| Error message | What it means | Fix |
|---|---|---|
| `arm_neon.h: No such file or directory` | Cross-compiler not finding ARM headers | Make sure toolchain file is pointing to the right compiler. Re-run cmake with `-DCMAKE_TOOLCHAIN_FILE=` |
| `error while loading shared libraries` | QEMU can't find ARM libc | Run: `export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu` |
| `ELF 64-bit ... x86-64` on `file test_mathext` | Binary compiled for x86, not ARM | The toolchain file wasn't used. Check your cmake command has `-DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-aarch64.cmake` |
| `CMake Error: could not find CMAKE_C_COMPILER` | Cross-compiler not installed | Re-run the apt install command from Step 1 |
| `Segmentation fault` in test | Bug in the implementation | Run with `qemu-aarch64 -L /usr/aarch64-linux-gnu ./test_mathext` directly for verbose output |

---

## Now Put It on GitHub (Team Sync)

Once one person has this working, push it so the whole team can clone it and verify
they can also build and pass the test.

```bash
# From armv8-lib/ root (not inside build/)
git init
git add .
git commit -m "feat: first working ARM function with test (vec_add_f32)"
```

On GitHub: create a new repository called `armv8-lib`, then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/armv8-lib.git
git branch -M main
git push -u origin main
```

Add a `.gitignore` first so you don't push compiled files:
```
build/
*.o
*.a
```

**Everyone else on the team:**
```bash
git clone https://github.com/YOUR_USERNAME/armv8-lib.git
cd armv8-lib
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-aarch64.cmake
cmake --build . -j$(nproc)
ctest --output-on-failure
```

All four of you seeing `100% tests passed` = you have a working shared baseline.
That is your 1. Everything in the previous guide — CI, 10 modules, branch strategy —
is built on top of this, one step at a time.

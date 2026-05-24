# navexa_ARMv8-A_Library

## Commands to run to setup
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

---

## Branch namings for better understanding
```bash
feat/<module>-<short-description>
fix/<module>-<bug-name>
docs/<what>
test/<module>-<what>
```
Example
```test
feat/sve-neon-fallback
feat/mte-bounds-checking
fix/crypto-aes-endianness
test/rng-statistical-tests
```
--- 
## Work flow for everyone
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
---
## Commit message syntax
```text
type(module): short description

Examples:
feat(sve): add 128-bit vector add with NEON fallback
fix(crypto): correct AES key schedule endianness
test(mte): add use-after-free detection test
docs(rng): document RNDR fallback behavior
refactor(atomics): reduce branching in cmpxchg path
```
---
## Further after code, The Testing part
Run this to download the required C++ testing unit
```bash
sudo apt install -y libgtest-dev
```
Command to run tests
```bash
cd build
ctest --output-on-failure
```


# Individual Workflow Guide — navexa Library
## AI-Assisted, 20 Hours, 2 Modules Per Person

---

## Before Anything Else — Your AI Setup (All Free)

You will use three AIs for different jobs. Do not use just one for everything.

| AI | Where | Use it for |
|---|---|---|
| **Grok** (grok.com) | Free, no login needed | Learning concepts, "explain this to me", understanding ARM docs |
| **Claude** (claude.ai) | Free tier | Writing and reviewing code, debugging, CMake help |
| **ChatGPT** (chatgpt.com) | Free tier | Second opinion on code, alternative explanations when stuck |

**The rule:** Grok = understand. Claude = build. ChatGPT = unstuck.

Set all three open in browser tabs before you start working. You will switch between them constantly.

---

## The 8 Modules and Who Owns What

Assign these in your team meeting. Each person picks 2.

| Module | Complexity | Good for |
|---|---|---|
| 1. SVE/NEON Emulation | High | Intermediate+ |
| 2. SME Matrix (GEMM) | High | Intermediate+ |
| 3. MTE Simulation | Medium | Intermediate |
| 4. Crypto (AES/SHA) | Medium | Intermediate |
| 5. Atomics | Medium | Intermediate |
| 6. RNG (RNDR + fallback) | Low-Medium | Beginner-friendly |
| 7. Virtualization Helpers | Medium-High | Intermediate |
| 8. Math/Transcendentals | Medium | Beginner-friendly |

**Suggested pairing for beginners:** Give them modules 6 + 8. They are self-contained,
have clear inputs/outputs, and don't require deep ARM architecture knowledge upfront.

---

## Hour-by-Hour Plan (20 Hours Total)

This is per person. Everyone follows the same structure, just on their own modules.

```
Hours 1–2   → Environment + Repo setup (everyone together)
Hours 3–4   → Learn your 2 modules (Grok-assisted reading)
Hours 5–6   → Write your headers (function signatures only, no implementation)
Hours 7–10  → Implement module 1 (AI-assisted coding)
Hours 11–12 → Write and pass tests for module 1
Hours 13–16 → Implement module 2 (AI-assisted coding)
Hours 17–18 → Write and pass tests for module 2
Hours 19–20 → Integration — merge PRs, fix conflicts, full test suite green
```

Do not skip the learning block (hours 3–4). People who jump straight to coding
without understanding what they're building produce code they can't debug.

---

## Part 1 — GitHub for People Who Only Know Add/Commit/Push

You know the basics. Here's everything on top of that which this project needs.

### Cloning the repo (first time)

```bash
git clone hhttps://github.com/adarsh-v-h/navexa_ARMv8-A_Library.git 
cd navexa
```

### The branch workflow — every single time you start working

Never work on `main` directly. Always:

```bash
# Step 1: Make sure your local main is up to date
git checkout main
git pull origin main

# Step 2: Create your feature branch
git checkout -b feat/rng-rndr-implementation

# Step 3: Do your work, make commits as you go
git add src/rng/rng.cpp tests/unit/test_rng.cpp
git commit -m "feat(rng): add RNDR hardware instruction with fallback"

# Step 4: Push your branch to GitHub
git push origin feat/rng-rndr-implementation

# Step 5: Go to GitHub website → you'll see a banner "Compare & pull request"
# Click it → write a short description → assign one teammate as reviewer → Create PR
```

### Keeping your branch up to date with main

If someone else merged their code while you were working, pull it in:

```bash
git fetch origin
git rebase origin/main
```

If rebase shows conflicts (it'll say "CONFLICT"), open the file it mentions,
look for the `<<<<<<<` markers, manually fix the conflict, then:

```bash
git add <the conflicted file>
git rebase --continue
```

### How to review someone else's PR

On GitHub, go to their PR → Files changed tab → read the code →
click the green "Review changes" button → write a short comment → Approve or Request changes.

You are not looking for perfection. You are checking:
- Does the header in `include/armv8lib/` match what the implementation actually does?
- Does the test actually test the function?
- Does it compile? (CI will tell you this automatically)

---

## Part 2 — How Headers, Source Files, and Tests Connect

This confuses beginners. Here's the mental model.

```
include/armv8lib/rng.h        ← The PROMISE. "I have a function called rng_generate."
src/rng/rng.cpp               ← The FULFILLMENT. The actual code that runs.
tests/unit/test_rng.cpp       ← The VERIFICATION. "Did the promise get fulfilled correctly?"
```

The header is what everyone else on the team sees and uses.
The .cpp is yours — nobody else touches it.
The test is also yours — you write it alongside your implementation.

### Header file rules

Every public function your module exposes goes in the header. Nothing else.
No implementation code. No `#include <arm_neon.h>` in headers (put that in your .cpp).

```cpp
// include/armv8lib/rng.h
#pragma once
#include <stdint.h>
#include <stdbool.h>

namespace navexa {
namespace rng {

/**
 * Generate a 64-bit random number.
 * Uses RNDR hardware instruction if available, ChaCha fallback otherwise.
 * @param out  Pointer to store result. Must not be NULL.
 * @return true if hardware RNG was used, false if software fallback was used.
 */
bool generate(uint64_t* out);

/**
 * Seed the software fallback PRNG.
 * Has no effect if hardware RNG is available.
 */
void seed_fallback(uint64_t seed_value);

} // namespace rng
} // namespace navexa
```

### Adding your module to CMakeLists.txt

After you create your .cpp file, add it to the library in `CMakeLists.txt`:

```cmake
add_library(navexa STATIC
    src/mathext/mathext.cpp
    src/rng/rng.cpp          # ← add your file here
)
```

And add your test:

```cmake
add_executable(test_rng tests/unit/test_rng.cpp)
target_link_libraries(test_rng navexa GTest::gtest_main)
target_compile_options(test_rng PRIVATE -march=armv8-a)
add_test(NAME rng COMMAND qemu-aarch64 -L /usr/aarch64-linux-gnu $<TARGET_FILE:test_rng>)
```

---

## Part 3 — Grok Prompts for Learning Your Module (Hours 3–4)

Open grok.com. Use these prompts in order for whichever module you own.
Don't skim — read each response and ask follow-ups until you actually understand it.

### Universal starter (use this first, replace MODULE_NAME)

```
I'm a C++ developer building a library for ARMv8-A architecture.
I need to understand [MODULE_NAME] well enough to implement it.

Please explain:
1. What problem does this solve? Give me a real-world example.
2. What does ARMv8-A provide natively for this, and what's missing?
3. What are the 3-5 most important concepts I need to understand before writing code?
4. Show me the simplest possible working example in C++.

Keep the explanation technical but assume I haven't worked with ARM before.
```

### Module-specific learning prompts

**Module 1 — SVE/NEON:**
```
Explain ARM NEON SIMD to me like I understand arrays and loops but have never
done SIMD. What is a float32x4_t? What does vaddq_f32 actually do in hardware?
Show me a before/after: the same operation as a normal loop, then as NEON code.
Then explain what SVE adds on top of NEON and why it matters.
```

**Module 2 — SME/GEMM:**
```
Explain matrix multiplication (GEMM) in C++ from scratch, then show me how
ARM SME accelerates it. What is an outer product? What does "streaming mode" mean
in the context of ARMv8/9? Show a naive GEMM implementation first, then an
optimized version using NEON tiling.
```

**Module 3 — MTE:**
```
What is a use-after-free bug and a buffer overflow? Show me a C++ code example
of each. Then explain how ARM MTE (Memory Tagging Extension) detects them at
hardware level. Since we're emulating MTE in software, what's the minimum
data structure needed to track tags on memory allocations?
```

**Module 4 — Crypto (AES/SHA):**
```
Explain AES-128 encryption at a conceptual level — what goes in, what comes out,
what is a "key schedule"? Then show me what ARM's AES crypto intrinsics look like
(vaeseq_u8, vaesmcq_u8) and how they map to the AES algorithm steps.
What does "constant-time" mean and why does it matter for crypto code?
```

**Module 5 — Atomics:**
```
What is a race condition? Show me a C++ example of one. Then explain what atomic
operations are and how they prevent it. What does acquire/release memory ordering
mean? What does ARMv8.1 add with LSE (Large System Extensions) atomics vs the
base ARMv8 load-exclusive/store-exclusive approach?
```

**Module 6 — RNG:**
```
What is the difference between a PRNG and a TRNG? Why does cryptographic code
need high-quality randomness? What is the RNDR instruction in ARMv8.5 and how
do you call it from C++? Show me a simple ChaCha20-based software fallback
for when hardware RNG isn't available.
```

**Module 7 — Virtualization:**
```
What is a hypervisor? What is VHE (Virtualization Host Extensions) in ARMv8?
What does "VM context switch" mean at the register level? I'm building a helper
library — what are the 3-4 most useful things a library can provide to someone
writing a hypervisor or VM monitor on ARMv8?
```

**Module 8 — Math/Transcendentals:**
```
What does "vectorized math" mean? Why is computing sin() or exp() slow on a CPU
and how does NEON help? Show me a NEON-optimized approximate exp() using the
Taylor series or a lookup table approach. What's the tradeoff between precision
and performance for vectorized transcendentals?
```

### When you're confused by a Grok response

```
That explanation lost me at [SPECIFIC PART]. Can you re-explain just that part
with a concrete numerical example? Walk through it step by step with actual
numbers, not variable names.
```

---

## Part 4 — Claude Prompts for Writing Code (Hours 5–16)

Claude is best for writing the actual implementation. Use these prompt patterns.

### Writing your header (Hour 5–6)

```
I'm building a C++ ARMv8-A library called navexa. I need to write the public
header for the [MODULE] module.

The module should expose these capabilities: [LIST FROM YOUR SKILLS.md SECTION]

Rules:
- Namespace is navexa::[module_name]
- All functions need a Doxygen comment explaining parameters and return value
- No implementation in the header — signatures only
- No ARM-specific headers in the header file (those go in the .cpp)
- Use #pragma once

Write the complete header file: include/navexa/[module].h
```

### Writing your implementation (Hours 7–10 and 13–16)

```
I'm implementing the [FUNCTION_NAME] function for a C++ ARMv8-A library.

Header signature:
[PASTE YOUR HEADER HERE]

Requirements:
- Use ARM NEON intrinsics where possible (arm_neon.h)
- Include a scalar fallback path for elements that don't fit in vectors
- Add inline comments explaining what each intrinsic does
- Target: aarch64-linux-gnu-g++ with -march=armv8-a

Write the complete .cpp implementation file.
```

### When the code doesn't compile

```
This C++ code for an ARMv8-A library is giving me a compile error.

Compiler: aarch64-linux-gnu-g++ with -march=armv8-a
Error message:
[PASTE THE FULL ERROR]

Code:
[PASTE YOUR CODE]

What's wrong and how do I fix it?
```

### When it compiles but the test fails

```
My ARMv8-A function compiles and runs via QEMU but the test is failing.

Function being tested:
[PASTE IMPLEMENTATION]

Test code:
[PASTE TEST]

Test output:
[PASTE WHAT CTEST PRINTED]

Walk me through what's wrong step by step.
```

### Writing your tests

```
I've implemented this function in a C++ ARMv8-A library:
[PASTE YOUR HEADER + IMPLEMENTATION]

Write Google Test unit tests for it. Include:
1. A normal case with known input/output
2. An edge case (zero, empty, boundary values)
3. A test that would catch an off-by-one error

Use EXPECT_ not ASSERT_ unless you explain why ASSERT_ is needed.
Add a comment on each test explaining what it's verifying.
```

---

## Part 5 — Git Prompts for Claude (For People Who Get Stuck)

When Git does something unexpected, paste this to Claude:

```
I'm working on a C++ GitHub project. I ran [GIT COMMAND] and got this output:
[PASTE OUTPUT]

I was trying to [WHAT YOU WERE TRYING TO DO].
What happened and what should I run next?
```

Common situations and what to ask:

**Merge conflict:**
```
I ran git rebase origin/main and got a conflict in [FILENAME].
Here's what the file looks like now: [PASTE FILE WITH <<< MARKERS]
My changes were: [DESCRIBE]
Their changes were: [DESCRIBE]
How do I resolve this correctly?
```

**Accidentally committed to main:**
```
I accidentally committed directly to main instead of a feature branch.
I haven't pushed yet. How do I move this commit to a new branch called
feat/rng-implementation and leave main clean?
```

**Pushed something wrong:**
```
I pushed a commit to my feature branch that I want to undo.
The commit message was "[MESSAGE]". I haven't opened a PR yet.
How do I remove it without breaking anything?
```

---

## Part 6 — Finding Resources to Learn (Grok-Powered)

You don't have time to read textbooks. Use Grok to find exactly what you need.

### Finding the right ARM documentation

```
I need to implement [FEATURE] for ARMv8-A. What specific sections of the
ARM Architecture Reference Manual should I read? Give me the section names
and what I'll find there. I don't need to read the whole thing — just what's
relevant to my implementation.
```

### Finding code references

```
Are there open-source C or C++ projects that implement [FEATURE] for ARMv8-A?
What are the best ones to read for reference? What files specifically should
I look at in each project?
```

Good reference projects to know about:
- `sse2neon` — x86 SSE to ARM NEON translation, very readable
- `ARM Compute Library` (github.com/ARM-software/ComputeLibrary) — production ARM code
- `OpenSSL` — crypto implementations including ARM intrinsics
- `highway` (google/highway) — portable SIMD, has ARM paths

### When you're learning a concept and Grok's explanation isn't clicking

```
I've read three explanations of [CONCEPT] and I still don't get it.
Here's what I think I understand so far: [YOUR CURRENT UNDERSTANDING]
Here's what's confusing me: [SPECIFIC CONFUSION]
Can you explain it differently, using an analogy to [SOMETHING FAMILIAR — like arrays, or file I/O, or cooking]?
```

---

## Part 7 — Daily Checklist for Each Work Session

Before you start:
```bash
git checkout main
git pull origin main
git checkout feat/your-branch  # switch back to your branch
git rebase origin/main         # pull in any new changes from teammates
```

While working — commit often, at least every hour:
```bash
git add src/yourmodule/yourfile.cpp
git commit -m "feat(module): short description of what you just did"
```

Before ending your session:
```bash
# Build and make sure nothing is broken
cd build
cmake --build . -j$(nproc)
ctest --output-on-failure

# Push your work
git push origin feat/your-branch
```

Never end a session with uncommitted work sitting locally. If you're mid-way
through something, commit it with `wip: ` prefix:
```bash
git commit -m "wip(rng): halfway through ChaCha fallback, not compiling yet"
```

---

## Part 8 — The 3 Rules That Keep the Team From Breaking Each Other

1. **Never touch a file you don't own.**
   If you need something from another module, ask the owner to add it to their header.
   Don't reach into their .cpp.

2. **Header changes need a message to the team.**
   If you change a function signature in your header after others have started using it,
   tell everyone immediately in your group chat. One signature change can break 3 builds.

3. **Green CI before you ask for review.**
   Don't open a PR if the CI is failing. Fix it first. A PR with a red CI
   gets ignored and creates noise.

---

## Quick Reference — Who to Ask When Stuck

| Problem | Ask |
|---|---|
| Don't understand a concept | Grok |
| Need code written or fixed | Claude |
| Code is logically wrong but compiles | Claude with full context |
| Need a second opinion on an approach | ChatGPT |
| Git confusion | Claude (paste exact error) |
| ARM intrinsic doesn't exist / wrong type | Grok ("what is the correct NEON intrinsic for X") |
| Teammate's code broke yours | Talk to them first, then Claude if needed |

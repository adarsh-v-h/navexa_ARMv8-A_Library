/*
 * tests/unit/test_sve.cpp
 * Unit tests for navexa::sve — SVE emulation layer
 *
 * What we test:
 *   - Predicate construction (pred_all, pred_first_n, pred_while_lt)
 *   - Predicated load/store (partial array tails)
 *   - Predicated arithmetic (merging semantics — inactive lanes keep 'a')
 *   - FMA correctness
 *   - Horizontal reduction (add, max) with active/inactive lane mix
 *   - Gather/scatter with non-contiguous indices
 *   - vecop_f32 end-to-end (including non-multiple-of-4 length)
 *   - f64 operations
 */

#include <gtest/gtest.h>
#include "armv8lib/sve.h"

#include <arm_neon.h>
#include <cmath>
#include <cstring>

using namespace navexa::sve;

/* -------------------------------------------------------------------------
 * Helper: extract float32x4_t into a plain array for assertions
 * ------------------------------------------------------------------------- */
static void extract(const float32x4_t v, float out[4]) {
    vst1q_f32(out, v);
}

static void extract64(const float64x2_t v, double out[2]) {
    vst1q_f64(out, v);
}

/* =========================================================================
 * Predicate construction tests
 * ========================================================================= */

TEST(SvePred, AllF32HasAllLanesActive) {
    SvePred pg = pred_all_f32();
    /* All 4 lanes must be active */
    EXPECT_TRUE(pg.active(0));
    EXPECT_TRUE(pg.active(1));
    EXPECT_TRUE(pg.active(2));
    EXPECT_TRUE(pg.active(3));
}

TEST(SvePred, AllF64HasTwoLanesActive) {
    SvePred pg = pred_all_f64();
    EXPECT_TRUE(pg.active(0));
    EXPECT_TRUE(pg.active(1));
    /* Lane 2 and 3 are irrelevant for f64 — not testing them */
}

TEST(SvePred, FirstN_Zero_NoLanesActive) {
    SvePred pg = pred_first_n(0);
    EXPECT_FALSE(pg.active(0));
    EXPECT_FALSE(pg.active(1));
    EXPECT_FALSE(pg.active(2));
    EXPECT_FALSE(pg.active(3));
}

TEST(SvePred, FirstN_Two_OnlyFirstTwoActive) {
    SvePred pg = pred_first_n(2);
    EXPECT_TRUE(pg.active(0));
    EXPECT_TRUE(pg.active(1));
    EXPECT_FALSE(pg.active(2));
    EXPECT_FALSE(pg.active(3));
}

TEST(SvePred, FirstN_Four_AllActive) {
    SvePred pg = pred_first_n(4);
    EXPECT_TRUE(pg.active(0));
    EXPECT_TRUE(pg.active(1));
    EXPECT_TRUE(pg.active(2));
    EXPECT_TRUE(pg.active(3));
}

TEST(SvePred, WhileLt_FullVector_AllActive) {
    /* idx=0, limit=4: all 4 lanes (0,1,2,3) are < 4 */
    SvePred pg = pred_while_lt(0, 4);
    EXPECT_TRUE(pg.active(0));
    EXPECT_TRUE(pg.active(1));
    EXPECT_TRUE(pg.active(2));
    EXPECT_TRUE(pg.active(3));
}

TEST(SvePred, WhileLt_TailOf1_OnlyLane0Active) {
    /* idx=4, limit=5: only lane 0 (index 4) is < 5 */
    SvePred pg = pred_while_lt(4, 5);
    EXPECT_TRUE(pg.active(0));
    EXPECT_FALSE(pg.active(1));
    EXPECT_FALSE(pg.active(2));
    EXPECT_FALSE(pg.active(3));
}

TEST(SvePred, WhileLt_TailOf3) {
    /* idx=5, limit=8: lanes 0,1,2 (indices 5,6,7) are < 8; lane 3 (index 8) is not */
    SvePred pg = pred_while_lt(5, 8);
    EXPECT_TRUE(pg.active(0));
    EXPECT_TRUE(pg.active(1));
    EXPECT_TRUE(pg.active(2));
    EXPECT_FALSE(pg.active(3));
}

TEST(SvePred, WhileLt_PastEnd_NoneActive) {
    /* idx=10, limit=8: no lanes active */
    SvePred pg = pred_while_lt(10, 8);
    EXPECT_FALSE(pg.active(0));
    EXPECT_FALSE(pg.active(1));
}

/* =========================================================================
 * Predicated load tests
 * ========================================================================= */

TEST(SveLoad, LoadAllLanes_ReadsAllFour) {
    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    SvePred pg   = pred_all_f32();
    float32x4_t v = load_f32(src, pg);
    float out[4];
    extract(v, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);
}

TEST(SveLoad, LoadPartialLanes_InactiveLanesAreZero) {
    /* Only lanes 0 and 1 active — src only has 2 valid elements */
    float src[2] = {7.0f, 8.0f};
    SvePred pg   = pred_first_n(2);
    float32x4_t v = load_f32(src, pg);
    float out[4];
    extract(v, out);
    EXPECT_FLOAT_EQ(out[0], 7.0f);
    EXPECT_FLOAT_EQ(out[1], 8.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);  /* inactive lane — must be 0 */
    EXPECT_FLOAT_EQ(out[3], 0.0f);  /* inactive lane — must be 0 */
}

/* =========================================================================
 * Predicated store tests
 * ========================================================================= */

TEST(SveStore, StoreAllLanes_WritesAllFour) {
    float32x4_t v  = vdupq_n_f32(99.0f);
    float dst[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
    store_f32(dst, v, pred_all_f32());
    EXPECT_FLOAT_EQ(dst[0], 99.0f);
    EXPECT_FLOAT_EQ(dst[1], 99.0f);
    EXPECT_FLOAT_EQ(dst[2], 99.0f);
    EXPECT_FLOAT_EQ(dst[3], 99.0f);
}

TEST(SveStore, StorePartialLanes_InactiveLanesUnchanged) {
    /* Store only lanes 0 and 1. Lanes 2 and 3 in dst must be untouched. */
    float vals[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float32x4_t v = vld1q_f32(vals);
    float dst[4]  = {-1.0f, -1.0f, -1.0f, -1.0f};
    store_f32(dst, v, pred_first_n(2));
    EXPECT_FLOAT_EQ(dst[0],  5.0f);  /* written */
    EXPECT_FLOAT_EQ(dst[1],  6.0f);  /* written */
    EXPECT_FLOAT_EQ(dst[2], -1.0f);  /* must NOT be overwritten */
    EXPECT_FLOAT_EQ(dst[3], -1.0f);  /* must NOT be overwritten */
}

/* =========================================================================
 * Predicated arithmetic — merging behaviour
 * ========================================================================= */

TEST(SveArith, AddF32_AllActive_AllLanesUpdated) {
    float av[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float bv[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float32x4_t a   = vld1q_f32(av);
    float32x4_t b   = vld1q_f32(bv);
    float32x4_t res = add_f32(a, b, pred_all_f32());
    float out[4];
    extract(res, out);
    EXPECT_FLOAT_EQ(out[0], 11.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 33.0f);
    EXPECT_FLOAT_EQ(out[3], 44.0f);
}

TEST(SveArith, AddF32_PartialActive_InactiveLanesKeepA) {
    /*
     * Merging predication: active lanes get a+b, inactive lanes keep 'a'.
     * Only lanes 0 and 2 active.
     */
    float av[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float bv[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float32x4_t a = vld1q_f32(av);
    float32x4_t b = vld1q_f32(bv);
    SvePred pg;
    pg.mask = 0b0101;  /* lanes 0 and 2 active */
    float32x4_t res = add_f32(a, b, pg);
    float out[4];
    extract(res, out);
    EXPECT_FLOAT_EQ(out[0], 11.0f);  /* active: 1+10 */
    EXPECT_FLOAT_EQ(out[1],  2.0f);  /* inactive: keeps a[1]=2 */
    EXPECT_FLOAT_EQ(out[2], 33.0f);  /* active: 3+30 */
    EXPECT_FLOAT_EQ(out[3],  4.0f);  /* inactive: keeps a[3]=4 */
}

TEST(SveArith, SubF32_AllActive) {
    float av[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float bv[4] = {1.0f,  2.0f,  3.0f,  4.0f};
    float32x4_t res = sub_f32(vld1q_f32(av), vld1q_f32(bv), pred_all_f32());
    float out[4];
    extract(res, out);
    EXPECT_FLOAT_EQ(out[0], 9.0f);
    EXPECT_FLOAT_EQ(out[1], 18.0f);
    EXPECT_FLOAT_EQ(out[2], 27.0f);
    EXPECT_FLOAT_EQ(out[3], 36.0f);
}

TEST(SveArith, MulF32_AllActive) {
    float av[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float bv[4] = {10.0f, 10.0f, 10.0f, 10.0f};
    float32x4_t res = mul_f32(vld1q_f32(av), vld1q_f32(bv), pred_all_f32());
    float out[4];
    extract(res, out);
    EXPECT_FLOAT_EQ(out[0], 20.0f);
    EXPECT_FLOAT_EQ(out[1], 30.0f);
    EXPECT_FLOAT_EQ(out[2], 40.0f);
    EXPECT_FLOAT_EQ(out[3], 50.0f);
}

TEST(SveArith, FmaF32_ResultIsATimesB_PlusC) {
    /*
     * fma(a, b, c) = a*b + c
     * a=[2,2,2,2], b=[3,3,3,3], c=[1,1,1,1] → result should be [7,7,7,7]
     */
    float32x4_t a = vdupq_n_f32(2.0f);
    float32x4_t b = vdupq_n_f32(3.0f);
    float32x4_t c = vdupq_n_f32(1.0f);
    float32x4_t res = fma_f32(a, b, c, pred_all_f32());
    float out[4];
    extract(res, out);
    EXPECT_FLOAT_EQ(out[0], 7.0f);
    EXPECT_FLOAT_EQ(out[1], 7.0f);
    EXPECT_FLOAT_EQ(out[2], 7.0f);
    EXPECT_FLOAT_EQ(out[3], 7.0f);
}

TEST(SveArith, FmaF32_InactiveLanesKeepA) {
    float32x4_t a = vdupq_n_f32(99.0f);
    float32x4_t b = vdupq_n_f32(2.0f);
    float32x4_t c = vdupq_n_f32(0.0f);
    SvePred pg;
    pg.mask = 0b0011;  /* only lanes 0 and 1 active */
    float32x4_t res = fma_f32(a, b, c, pg);
    float out[4];
    extract(res, out);
    EXPECT_FLOAT_EQ(out[0], 198.0f); /* 99*2+0 */
    EXPECT_FLOAT_EQ(out[1], 198.0f);
    EXPECT_FLOAT_EQ(out[2],  99.0f); /* inactive: keeps a */
    EXPECT_FLOAT_EQ(out[3],  99.0f); /* inactive: keeps a */
}

/* =========================================================================
 * Horizontal reduction tests
 * ========================================================================= */

TEST(SveReduce, ReduceAddF32_AllLanes) {
    float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float32x4_t v = vld1q_f32(vals);
    float result  = reduce_add_f32(v, pred_all_f32());
    EXPECT_FLOAT_EQ(result, 10.0f);  /* 1+2+3+4 */
}

TEST(SveReduce, ReduceAddF32_PartialLanes) {
    /* Only lanes 0 and 1 active — sum should be 1+2=3, not 1+2+3+4=10 */
    float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float32x4_t v = vld1q_f32(vals);
    float result  = reduce_add_f32(v, pred_first_n(2));
    EXPECT_FLOAT_EQ(result, 3.0f);
}

TEST(SveReduce, ReduceMaxF32_AllLanes) {
    float vals[4] = {3.0f, 1.0f, 4.0f, 2.0f};
    float32x4_t v = vld1q_f32(vals);
    float result  = reduce_max_f32(v, pred_all_f32());
    EXPECT_FLOAT_EQ(result, 4.0f);
}

TEST(SveReduce, ReduceMaxF32_ExcludesInactiveLanes) {
    /* Lane 2 has the global max (99) but it's inactive — should not win */
    float vals[4] = {3.0f, 7.0f, 99.0f, 2.0f};
    float32x4_t v = vld1q_f32(vals);
    SvePred pg;
    pg.mask = 0b1011;  /* lanes 0,1,3 active; lane 2 inactive */
    float result  = reduce_max_f32(v, pg);
    EXPECT_FLOAT_EQ(result, 7.0f);
}

/* =========================================================================
 * Gather / Scatter tests
 * ========================================================================= */

TEST(SveGather, GatherF32_NonContiguousIndices) {
    float base[8] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    int32_t idx[4] = {7, 0, 5, 2};  /* gather indices 7,0,5,2 */
    SvePred pg = pred_all_f32();
    float32x4_t v = gather_f32(base, idx, pg);
    float out[4];
    extract(v, out);
    EXPECT_FLOAT_EQ(out[0], 80.0f);  /* base[7] */
    EXPECT_FLOAT_EQ(out[1], 10.0f);  /* base[0] */
    EXPECT_FLOAT_EQ(out[2], 60.0f);  /* base[5] */
    EXPECT_FLOAT_EQ(out[3], 30.0f);  /* base[2] */
}

TEST(SveGather, GatherF32_InactiveLanesLoadZero) {
    float base[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t idx[4] = {0, 1, 2, 3};
    SvePred pg = pred_first_n(2);  /* only lanes 0 and 1 active */
    float32x4_t v = gather_f32(base, idx, pg);
    float out[4];
    extract(v, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);   /* active */
    EXPECT_FLOAT_EQ(out[1], 2.0f);   /* active */
    EXPECT_FLOAT_EQ(out[2], 0.0f);   /* inactive — must be zero */
    EXPECT_FLOAT_EQ(out[3], 0.0f);   /* inactive — must be zero */
}

TEST(SveScatter, ScatterF32_WritesToCorrectPositions) {
    float base[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int32_t idx[4] = {1, 3, 5, 7};
    float vals[4]  = {100.0f, 200.0f, 300.0f, 400.0f};
    float32x4_t v  = vld1q_f32(vals);
    scatter_f32(base, idx, v, pred_all_f32());
    EXPECT_FLOAT_EQ(base[1], 100.0f);
    EXPECT_FLOAT_EQ(base[3], 200.0f);
    EXPECT_FLOAT_EQ(base[5], 300.0f);
    EXPECT_FLOAT_EQ(base[7], 400.0f);
    /* Unwritten positions must stay zero */
    EXPECT_FLOAT_EQ(base[0], 0.0f);
    EXPECT_FLOAT_EQ(base[2], 0.0f);
}

TEST(SveScatter, ScatterF32_InactiveLanesDoNotWrite) {
    float base[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    int32_t idx[4] = {0, 1, 2, 3};
    float vals[4]  = {77.0f, 88.0f, 99.0f, 111.0f};
    float32x4_t v  = vld1q_f32(vals);
    SvePred pg;
    pg.mask = 0b0101;  /* lanes 0 and 2 only */
    scatter_f32(base, idx, v, pg);
    EXPECT_FLOAT_EQ(base[0],  77.0f);  /* written */
    EXPECT_FLOAT_EQ(base[1],  -1.0f);  /* inactive — untouched */
    EXPECT_FLOAT_EQ(base[2],  99.0f);  /* written */
    EXPECT_FLOAT_EQ(base[3],  -1.0f);  /* inactive — untouched */
}

/* =========================================================================
 * vecop_f32 end-to-end tests (the SVE loop pattern)
 * ========================================================================= */

TEST(SveVecop, AddF32_ExactMultipleOf4) {
    float a[8]   = {1,2,3,4,5,6,7,8};
    float b[8]   = {10,10,10,10,10,10,10,10};
    float out[8] = {};
    vecop_f32(out, a, b, 8, SveOp::ADD);
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(out[i], a[i] + b[i]) << "mismatch at i=" << i;
    }
}

TEST(SveVecop, AddF32_NonMultipleOf4_TailHandled) {
    /* len=6: first 4 via NEON, last 2 via predicated tail */
    float a[6]   = {1,2,3,4,5,6};
    float b[6]   = {10,20,30,40,50,60};
    float out[6] = {};
    vecop_f32(out, a, b, 6, SveOp::ADD);
    EXPECT_FLOAT_EQ(out[0], 11.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 33.0f);
    EXPECT_FLOAT_EQ(out[3], 44.0f);
    EXPECT_FLOAT_EQ(out[4], 55.0f);
    EXPECT_FLOAT_EQ(out[5], 66.0f);
}

TEST(SveVecop, MulF32_NonMultipleOf4_LenOf1) {
    float a[1]   = {5.0f};
    float b[1]   = {3.0f};
    float out[1] = {};
    vecop_f32(out, a, b, 1, SveOp::MUL);
    EXPECT_FLOAT_EQ(out[0], 15.0f);
}

TEST(SveVecop, SubF32_AllElements) {
    float a[4]   = {100.0f, 200.0f, 300.0f, 400.0f};
    float b[4]   = {1.0f,   2.0f,   3.0f,   4.0f};
    float out[4] = {};
    vecop_f32(out, a, b, 4, SveOp::SUB);
    EXPECT_FLOAT_EQ(out[0],  99.0f);
    EXPECT_FLOAT_EQ(out[1], 198.0f);
    EXPECT_FLOAT_EQ(out[2], 297.0f);
    EXPECT_FLOAT_EQ(out[3], 396.0f);
}

/* =========================================================================
 * f64 operation tests
 * ========================================================================= */

TEST(SveF64, AddF64_AllActive) {
    double av[2] = {1.5, 2.5};
    double bv[2] = {10.0, 20.0};
    float64x2_t a   = vld1q_f64(av);
    float64x2_t b   = vld1q_f64(bv);
    float64x2_t res = add_f64(a, b, pred_all_f64());
    double out[2];
    extract64(res, out);
    EXPECT_DOUBLE_EQ(out[0], 11.5);
    EXPECT_DOUBLE_EQ(out[1], 22.5);
}

TEST(SveF64, LoadStoreF64_PartialPredicate) {
    double src[1] = {3.14};
    float64x2_t v = load_f64(src, pred_first_n(1));  /* only lane 0 */
    double out[2];
    extract64(v, out);
    EXPECT_DOUBLE_EQ(out[0], 3.14);
    EXPECT_DOUBLE_EQ(out[1], 0.0); /* inactive lane = 0 */
}
// tests/unit/test_gemm.cpp
#include "armv8lib/sme/gemm.hpp"
#include <gtest/gtest.h>
using namespace armv8::sme;
static void expect_near(const Matrix<float>& r, const Matrix<float>& e, float tol=1e-4f){
    ASSERT_EQ(r.rows(),e.rows()); ASSERT_EQ(r.cols(),e.cols());
    for(std::size_t i=0;i<r.rows();++i)
        for(std::size_t j=0;j<r.cols();++j)
            EXPECT_NEAR(r(i,j),e(i,j),tol)<<"at ("<<i<<","<<j<<")";
}
TEST(GemmTest,Basic2x2){
    Matrix<float> A(2,2,{1,2,3,4}),B(2,2,{5,6,7,8}),exp(2,2,{19,22,43,50});
    expect_near(gemm_simple(A,B),exp);
}
TEST(GemmTest,NonSquare){
    Matrix<float> A(3,2,{1,2,3,4,5,6}),B(2,4,{1,0,1,0,0,1,0,1});
    Matrix<float> exp(3,4,{1,2,1,2,3,4,3,4,5,6,5,6});
    expect_near(gemm_simple(A,B),exp);
}
TEST(GemmTest,AlphaBeta){
    Matrix<float> A(2,2,{1,0,0,1}),B(2,2,{3,4,5,6}),C(2,2,{1,1,1,1});
    Matrix<float> exp(2,2,{9,11,13,15});
    gemm(A,B,C,2.0f,3.0f); expect_near(C,exp);
}
TEST(GemmTest,BetaZero){
    Matrix<float> A(2,2,{1,0,0,1}),B(2,2,{7,8,9,10}),C(2,2,999.0f);
    gemm(A,B,C,1.0f,0.0f); expect_near(C,B);
}
TEST(GemmTest,DimMismatch){
    Matrix<float> A(2,3),B(4,2),C(2,2);
    EXPECT_THROW(gemm(A,B,C),std::invalid_argument);
}
TEST(GemmTest,DoublePrecision){
    Matrix<double> A(2,2,{1,2,3,4}),B(2,2,{1,0,0,1});
    auto C=gemm_simple(A,B);
    for(std::size_t i=0;i<2;++i)
        for(std::size_t j=0;j<2;++j)
            EXPECT_DOUBLE_EQ(C(i,j),A(i,j));
}

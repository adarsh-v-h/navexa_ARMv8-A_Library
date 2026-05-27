#include "armv8lib/sme/gemm.hpp"
#include <stdexcept>
#include <algorithm>
namespace armv8 {
namespace sme {
namespace { constexpr std::size_t TILE = 64; }
template <typename T>
void gemm(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, T alpha, T beta)
{
    if(A.cols()!=B.rows()) throw std::invalid_argument("gemm: A.cols must equal B.rows");
    if(C.rows()!=A.rows()||C.cols()!=B.cols()) throw std::invalid_argument("gemm: C must be A.rows x B.cols");
    const std::size_t M=A.rows(),K=A.cols(),N=B.cols();
    if(beta==T{0}) C.zero();
    else if(beta!=T{1}) for(std::size_t i=0;i<M;++i) for(std::size_t j=0;j<N;++j) C(i,j)*=beta;
    for(std::size_t i0=0;i0<M;i0+=TILE){ const std::size_t ie=std::min(i0+TILE,M);
    for(std::size_t k0=0;k0<K;k0+=TILE){ const std::size_t ke=std::min(k0+TILE,K);
    for(std::size_t j0=0;j0<N;j0+=TILE){ const std::size_t je=std::min(j0+TILE,N);
        for(std::size_t i=i0;i<ie;++i)
        for(std::size_t k=k0;k<ke;++k){ const T a=alpha*A(i,k);
        for(std::size_t j=j0;j<je;++j) C(i,j)+=a*B(k,j); }}}}
}
template <typename T>
Matrix<T> gemm_simple(const Matrix<T>& A, const Matrix<T>& B)
{
    if(A.cols()!=B.rows()) throw std::invalid_argument("gemm_simple: A.cols must equal B.rows");
    Matrix<T> C(A.rows(),B.cols(),T{0});
    gemm(A,B,C,T{1},T{0});
    return C;
}
template void gemm<float>(const Matrix<float>&,const Matrix<float>&,Matrix<float>&,float,float);
template void gemm<double>(const Matrix<double>&,const Matrix<double>&,Matrix<double>&,double,double);
template void gemm<int>(const Matrix<int>&,const Matrix<int>&,Matrix<int>&,int,int);
template Matrix<float>  gemm_simple<float> (const Matrix<float>&, const Matrix<float>&);
template Matrix<double> gemm_simple<double>(const Matrix<double>&,const Matrix<double>&);
template Matrix<int>    gemm_simple<int>   (const Matrix<int>&,   const Matrix<int>&);
} // namespace sme
} // namespace armv8

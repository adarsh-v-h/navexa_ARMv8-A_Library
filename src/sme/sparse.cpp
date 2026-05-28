#include "armv8lib/sme/types.hpp"
#include "armv8lib/sme/sparse.hpp"
#include <stdexcept>
namespace armv8 {
namespace sme {
template <typename T>
void spmm(const SparseCSR<T>& A, const Matrix<T>& B, Matrix<T>& C, T alpha, T beta)
{
    if(A.cols()!=B.rows()) throw std::invalid_argument("spmm: A.cols must equal B.rows");
    if(C.rows()!=A.rows()||C.cols()!=B.cols()) throw std::invalid_argument("spmm: C must be A.rows x B.cols");
    if(beta==T{0}) C.zero();
    else if(beta!=T{1}) for(std::size_t i=0;i<C.rows();++i) for(std::size_t j=0;j<C.cols();++j) C(i,j)*=beta;
    for(std::size_t i=0;i<A.rows();++i){
        auto [rs,re]=A.row_range(i);
        for(std::size_t ptr=rs;ptr<re;++ptr){
            const T val=alpha*A.values()[ptr];
            const std::size_t k=A.col_idx()[ptr];
            for(std::size_t j=0;j<B.cols();++j) C(i,j)+=val*B(k,j);
        }
    }
}
template <typename T>
void spmv(const SparseCSR<T>& A, const std::vector<T>& x, std::vector<T>& y, T alpha, T beta)
{
    if(A.cols()!=x.size()) throw std::invalid_argument("spmv: A.cols must equal x.size()");
    if(A.rows()!=y.size()) throw std::invalid_argument("spmv: A.rows must equal y.size()");
    for(auto& yi:y) yi*=beta;
    for(std::size_t i=0;i<A.rows();++i){
        auto [rs,re]=A.row_range(i);
        T sum=T{0};
        for(std::size_t ptr=rs;ptr<re;++ptr) sum+=A.values()[ptr]*x[A.col_idx()[ptr]];
        y[i]+=alpha*sum;
    }
}
template void spmm<float> (const SparseCSR<float>&,  const Matrix<float>&,  Matrix<float>&,  float,  float);
template void spmm<double>(const SparseCSR<double>&, const Matrix<double>&, Matrix<double>&, double, double);
template void spmv<float> (const SparseCSR<float>&,  const std::vector<float>&,  std::vector<float>&,  float,  float);
template void spmv<double>(const SparseCSR<double>&, const std::vector<double>&, std::vector<double>&, double, double);
} // namespace sme
} // namespace armv8

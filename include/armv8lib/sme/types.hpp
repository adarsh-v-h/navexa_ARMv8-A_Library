#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace armv8 {
namespace sme {

template <typename T>
class Matrix {
    static_assert(std::is_arithmetic<T>::value, "Matrix<T>: T must be arithmetic");
public:
    using value_type = T;
    using size_type  = std::size_t;

    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_type r, size_type c) : rows_(r), cols_(c), data_(r*c, T{}) {}
    Matrix(size_type r, size_type c, T fill) : rows_(r), cols_(c), data_(r*c, fill) {}
    Matrix(size_type r, size_type c, std::initializer_list<T> v) : rows_(r), cols_(c), data_(v) {
        if(data_.size()!=r*c) throw std::invalid_argument("Matrix: init list size mismatch");
    }
    Matrix(const Matrix&)=default; Matrix& operator=(const Matrix&)=default;
    Matrix(Matrix&&)=default;      Matrix& operator=(Matrix&&)=default;

    T& at(size_type i,size_type j){ check(i,j); return data_[i*cols_+j]; }
    const T& at(size_type i,size_type j) const { check(i,j); return data_[i*cols_+j]; }
    T& operator()(size_type i,size_type j) noexcept { return data_[i*cols_+j]; }
    const T& operator()(size_type i,size_type j) const noexcept { return data_[i*cols_+j]; }
    T*       data()       noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }
    size_type rows()  const noexcept { return rows_; }
    size_type cols()  const noexcept { return cols_; }
    size_type size()  const noexcept { return rows_*cols_; }
    bool      empty() const noexcept { return rows_==0||cols_==0; }
    void fill(T v) noexcept { std::fill(data_.begin(),data_.end(),v); }
    void zero()    noexcept { fill(T{}); }
    bool operator==(const Matrix& o) const noexcept { return rows_==o.rows_&&cols_==o.cols_&&data_==o.data_; }
    bool operator!=(const Matrix& o) const noexcept { return !(*this==o); }

private:
    size_type rows_, cols_;
    std::vector<T> data_;
    void check(size_type i,size_type j) const {
        if(i>=rows_||j>=cols_) throw std::out_of_range("Matrix: index out of range");
    }
};

template <typename T>
class SparseCSR {
    static_assert(std::is_arithmetic<T>::value, "SparseCSR<T>: T must be arithmetic");
public:
    using value_type = T;
    using size_type  = std::size_t;

    SparseCSR() : rows_(0), cols_(0) {}
    SparseCSR(size_type rows, size_type cols,
              std::vector<T>         values,
              std::vector<size_type> col_idx,
              std::vector<size_type> row_ptr)
        : rows_(rows), cols_(cols),
          values_(std::move(values)),
          col_idx_(std::move(col_idx)),
          row_ptr_(std::move(row_ptr))
    {
        if(row_ptr_.size() != rows_+1)
            throw std::invalid_argument("SparseCSR: row_ptr size must be rows+1");
    }

    static SparseCSR from_dense(const Matrix<T>& m) {
        std::vector<T>         vals;
        std::vector<size_type> colidx, rowptr;
        rowptr.reserve(m.rows()+1);
        rowptr.push_back(0);
        for(size_type i=0;i<m.rows();++i){
            for(size_type j=0;j<m.cols();++j){
                if(m(i,j)!=T{}){
                    vals.push_back(m(i,j));
                    colidx.push_back(j);
                }
            }
            rowptr.push_back(static_cast<size_type>(vals.size()));
        }
        return SparseCSR(m.rows(), m.cols(),
                         std::move(vals), std::move(colidx), std::move(rowptr));
    }

    Matrix<T> to_dense() const {
        Matrix<T> m(rows_, cols_);
        for(size_type i=0;i<rows_;++i)
            for(size_type ptr=row_ptr_[i];ptr<row_ptr_[i+1];++ptr)
                m(i, col_idx_[ptr]) = values_[ptr];
        return m;
    }

    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }
    size_type nnz()  const noexcept { return values_.size(); }

    const std::vector<T>&         values()  const noexcept { return values_; }
    const std::vector<size_type>& col_idx() const noexcept { return col_idx_; }
    const std::vector<size_type>& row_ptr() const noexcept { return row_ptr_; }

    double sparsity() const noexcept {
        if(rows_==0||cols_==0) return 0.0;
        return 1.0 - static_cast<double>(values_.size()) / static_cast<double>(rows_*cols_);
    }

    std::pair<size_type,size_type> row_range(size_type i) const {
        return {row_ptr_[i], row_ptr_[i+1]};
    }

private:
    size_type              rows_, cols_;
    std::vector<T>         values_;
    std::vector<size_type> col_idx_;
    std::vector<size_type> row_ptr_;
};

} // namespace sme
} // namespace armv8

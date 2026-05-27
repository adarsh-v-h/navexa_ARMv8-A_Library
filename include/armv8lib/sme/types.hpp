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
} // namespace sme
} // namespace armv8

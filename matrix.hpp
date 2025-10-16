// matrix.hpp
#pragma once
#include <vector>
#include <cstddef>
#include <cassert>
#include <iostream>

class Matrix {
public:
    Matrix() = default;

    Matrix(size_t rows, size_t cols, float val = 0.0f);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Access elements (row, col)
    float& operator()(size_t r, size_t c);
    float  operator()(size_t r, size_t c) const;

    // Fill matrix with constant
    void fill(float v);

    // Static constructors
    static Matrix zeros(size_t r, size_t c);
    static Matrix ones(size_t r, size_t c);

    // Core operations
    static Matrix matmul(const Matrix& A, const Matrix& B);
    void add_rowwise(const std::vector<float>& b);
    void apply_relu();
    static Matrix softmax_cols(const Matrix& Z);
    void he_normal(unsigned seed = 42);
    Matrix transpose() const;
    Matrix slice_cols(size_t start, size_t end) const;
    // Print helper for debugging
    void print(size_t max_r = 5, size_t max_c = 5) const;

private:
    size_t rows_ = 0, cols_ = 0;
    std::vector<float> data_;
};

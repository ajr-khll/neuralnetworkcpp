#include <algorithm>
#include "matrix.hpp"
#include <cmath>
#include <random>

Matrix::Matrix(size_t rows, size_t cols, float val)
    : rows_(rows), cols_(cols), data_(rows * cols, val) {}



// Overload Matrix(r, c) to access and modify elements or just access for const cases
float& Matrix::operator()(size_t r, size_t c) {
    assert(r < rows_ && c < cols_);
    return data_[r * cols_ + c];
} // returns reference to allow modification

float Matrix::operator()(size_t r, size_t c) const {
    assert(r < rows_ && c < cols_);
    return data_[r * cols_ + c];
}

// fill matrix with any constant value v
void Matrix::fill(float v) {
    std::fill(data_.begin(), data_.end(), v);
}

// Static constructors for common matrices
Matrix Matrix::zeros(size_t r, size_t c) {
    return Matrix(r, c, 0.0f);
}

Matrix Matrix::ones(size_t r, size_t c) {
    return Matrix(r, c, 1.0f);
}

// Matrix multiplication: C = A * B
Matrix Matrix::matmul(const Matrix& A, const Matrix& B) {
    assert(A.cols() == B.rows());
    Matrix C(A.rows(), B.cols(), 0.0f);

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            float aik = A(i, k); // grab the value we're on for A
            for (size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += aik * B(k, j); // multiply the value from A with the 
            }
        }
    }
    return C;
}

// Add bias vector (row-wise)
void Matrix::add_rowwise(const std::vector<float>& b) {
    assert(b.size() == rows_);
    for (size_t r = 0; r < rows_; ++r)
        for (size_t c = 0; c < cols_; ++c)
            (*this)(r, c) += b[r];
}

// Apply ReLU activation function element-wise
void Matrix::apply_relu() {
    for (auto& val : data_)
        val = std::max(0.0f, val);
}

// Transpose the matrix
Matrix Matrix::transpose() const {
    Matrix T(cols_, rows_);
    for (size_t r = 0; r < rows_; ++r)
        for (size_t c = 0; c < cols_; ++c)
            T(c, r) = (*this)(r, c);
    return T;
}

Matrix Matrix::slice_cols(size_t start, size_t count) const {
    if (start + count > cols_) throw std::out_of_range("slice_cols out of range");
    Matrix result(rows_, count);
    for (size_t c = 0; c < count; ++c) {
        for (size_t r = 0; r < rows_; ++r) {
            result(r, c) = (*this)(r, start + c);
        }
    }
    return result;
}

void Matrix::he_normal(unsigned seed) {
    if (cols_ == 0) return; // nothing to initialize

    float stddev = std::sqrt(2.0f / static_cast<float>(cols_));

    // Mersenne Twister RNG seeded with given number
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (auto& w : data_) {
        w = dist(rng);
    }
}

// Softmax along columns (for each column vector)
Matrix Matrix::softmax_cols(const Matrix& Z) {
    Matrix S(Z.rows(), Z.cols());
    for (size_t c = 0; c < Z.cols(); ++c) {
        float max_val = Z(0, c);
        for (size_t r = 1; r < Z.rows(); ++r) {
            if (Z(r, c) > max_val) max_val = Z(r, c);
        }
        float sum_exp = 0.0f;
        for (size_t r = 0; r < Z.rows(); ++r) {
            S(r, c) = std::exp(Z(r, c) - max_val);
            sum_exp += S(r, c);
        }
        for (size_t r = 0; r < Z.rows(); ++r) {
            S(r, c) /= sum_exp;
        }
    }
    return S;
}

// Print a few elements for debugging
void Matrix::print(size_t max_r, size_t max_c) const {
    size_t R = std::min(rows_, max_r);
    size_t C = std::min(cols_, max_c);
    for (size_t r = 0; r < R; ++r) {
        for (size_t c = 0; c < C; ++c) {
            std::cout << (*this)(r, c) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "--- " << rows_ << "x" << cols_ << " ---\n";
}

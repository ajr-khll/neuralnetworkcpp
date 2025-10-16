#include <algorithm>
#include <vector>
#include "matrix.hpp"
#include <cmath>
#include "nn_utils.hpp"

std::vector<float> sum_cols(const Matrix& M) {
    std::vector<float> s(M.rows(), 0.0f);
    for (size_t r = 0; r < M.rows(); ++r) {
        for (size_t c = 0; c < M.cols(); ++c) {
            s[r] += M(r,c);
        }
    }
    return s;
}

void subtract_onehot_cols(Matrix& M, const std::vector<int>& y) {
    for (size_t i = 0; i < y.size(); ++i) {
        M(y[i], i) -= 1.0f;
    }
}

void scale_inplace(Matrix& M, float s) {
    for (size_t r = 0; r < M.rows(); ++r) {
        for (size_t c = 0; c < M.cols(); ++c) {
            M(r, c) *= s;
        }
    }
}

void add_scaled_inplace(Matrix& W, const Matrix& dW, float scale) {
    assert(W.rows() == dW.rows() && W.cols() == dW.cols());
    for (size_t r = 0; r < W.rows(); ++r)
        for (size_t c = 0; c < W.cols(); ++c)
            W(r, c) += scale * dW(r, c);
}

void add_scaled_inplace(std::vector<float>& b, const std::vector<float>& db, float scale) {
    assert(b.size() == db.size());
    for (size_t i = 0; i < b.size(); ++i)
        b[i] += scale * db[i];
}

void relu_backward_inplace(Matrix& dA, const Matrix& A1) {
    assert(dA.rows() == A1.rows() && dA.cols() == A1.cols());
    for (size_t r = 0; r < dA.rows(); ++r)
        for (size_t c = 0; c < dA.cols(); ++c)
            if (A1(r, c) <= 0.0f) dA(r, c) = 0.0f;
}

float accuracy_from_probs(const Matrix& P, const std::vector<int>& y) {
    size_t correct = 0;
    for (size_t c = 0; c < P.cols(); ++c) {
        // argmax row
        size_t best_r = 0;
        float best_v = P(0, c);
        for (size_t r = 1; r < P.rows(); ++r)
            if (P(r, c) > best_v) { best_v = P(r, c); best_r = r; }
        if ((int)best_r == y[c]) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(P.cols());
}

float mean_cross_entropy(const Matrix& P, const std::vector<int>& y) {
    const float eps = 1e-12f;
    float sum = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) {
        float p_true = P(y[i], i);
        if (p_true < eps) p_true = eps;
        sum += -std::log(p_true);
    }
    return sum / static_cast<float>(y.size());
}




StepResult train_step(
    Matrix& W1, std::vector<float>& b1,
    Matrix& W2, std::vector<float>& b2,
    const Matrix& X_batch,
    const std::vector<int>& y_batch,
    float lr
) {
    // Forward propagation
    const size_t B = X_batch.cols();
    Matrix Z1 = Matrix::matmul(W1, X_batch);
    Z1.add_rowwise(b1);
    Matrix A1 = Z1;
    A1.apply_relu();

    // Second pass
    Matrix Z2 = Matrix::matmul(W2, A1);
    Z2.add_rowwise(b2);
    Matrix P = Matrix::softmax_cols(Z2);

    // stats

    float loss = mean_cross_entropy(P, y_batch);
    float acc = accuracy_from_probs(P, y_batch);


    // Backpropagation
    //dZ2 = (P-onehot)/b
    Matrix dZ2 = P;
    subtract_onehot_cols(dZ2, y_batch);
    scale_inplace(dZ2, 1.0f/static_cast<float>(B));

    // dW2 = dZ2 * A1^T
    Matrix dW2 = Matrix::matmul(dZ2, A1.transpose());
    // db2 = sum_cols(dZ2)
    std::vector<float> db2 = sum_cols(dZ2);

    //dA1 = W2^T * dZ2
    Matrix dA1 = Matrix::matmul(W2.transpose(), dZ2);
    Matrix dZ1 = dA1;
    relu_backward_inplace(dZ1, A1); // relu in reverse literally


    // dW1 = dZ1 * X^T
    Matrix dW1 = Matrix::matmul(dZ1, X_batch.transpose());
    // db1 = sum_cols(dZ1)
    std::vector<float> db1 = sum_cols(dZ1);


    // ===== SGD UPDATE =====
    const float neg_lr = -lr;
    add_scaled_inplace(W2, dW2, neg_lr);
    add_scaled_inplace(b2, db2, neg_lr);
    add_scaled_inplace(W1, dW1, neg_lr);
    add_scaled_inplace(b1, db1, neg_lr);

    return {loss, acc};
}
#pragma once
#include "matrix.hpp"
#include <vector>

struct StepResult {
    float loss;
    float acc;
};

std::vector<float> sum_cols(const Matrix& M);
void subtract_onehot_cols(Matrix& M, const std::vector<int>& y);
void scale_inplace(Matrix& M, float s);
void add_scaled_inplace(Matrix& W, const Matrix& dW, float scale);
void add_scaled_inplace(std::vector<float>& b, const std::vector<float>& db, float scale);
void relu_backward_inplace(Matrix& dA, const Matrix& A1);
float accuracy_from_probs(const Matrix& P, const std::vector<int>& y);
float mean_cross_entropy(const Matrix& P, const std::vector<int>& y);

StepResult train_step(
    Matrix& W1, std::vector<float>& b1,
    Matrix& W2, std::vector<float>& b2,
    const Matrix& X_batch,
    const std::vector<int>& y_batch,
    float lr
);
#include <iostream>        // for std::cout
#include <vector>
#include <cmath>

#include "matrix.hpp"
#include "csvreader.hpp"
#include "nn_utils.hpp"

int main() {
    // 1) Load data
    auto train = csv_to_matrix("data/mnist_train.csv", /*has_header=*/true, /*normalize=*/true);

    // If your loader didn’t already return 784×N, transpose once:
    if (train.X.rows() != 784) {
        train.X = train.X.transpose();  // ensure X is 784 × N
    }

    // 2) Model sizes
    const int input_size  = 784;
    const int hidden_size = 128;
    const int output_size = 10;

    // 3) Parameters (weights + biases)
    Matrix W1(hidden_size, input_size);   // 128 × 784
    Matrix W2(output_size, hidden_size);  // 10  × 128
    std::vector<float> b1(hidden_size, 0.0f);
    std::vector<float> b2(output_size, 0.0f);

    // Init (He normal)
    W1.he_normal(42);
    W2.he_normal(1337);

    // 4) Training hyperparams
    const size_t B  = 128;    // batch size
    const float  lr = 1e-2f;  // learning rate

    // 5) One pass over the dataset (you can wrap this in an epoch loop later)
    const size_t N = train.X.cols();  // number of examples
    for (size_t i = 0; i + B <= N; i += B) {
        Matrix X_batch = train.X.slice_cols(i, B);                         // 784 × B
        std::vector<int> y_batch(train.y.begin() + i, train.y.begin() + i + B);

        StepResult r = train_step(W1, b1, W2, b2, X_batch, y_batch, lr);

        if ((i / B) % 50 == 0) {
            std::cout << "iter " << (i / B)
                      << "  loss " << r.loss
                      << "  acc "  << r.acc << "\n";
        }
    }

    return 0;
}


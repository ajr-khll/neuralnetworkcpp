#pragma once
#include <string>
#include <vector>
#include "matrix.hpp"

// Loads a Kaggle MNIST CSV (label, 784 pixels per row).
// Returns X with shape (784 x N) and y with N labels.
// has_header: skip first line if true.
// normalize: divide pixels by 255 if true.
struct CSVResult {
    Matrix X;
    std::vector<int> y;
};

CSVResult csv_to_matrix(const std::string& path,
                        bool has_header = true,
                        bool normalize  = true);

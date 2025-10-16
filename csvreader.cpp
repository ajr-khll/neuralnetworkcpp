#include "csvreader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

static std::size_t count_data_rows(const std::string& path, bool has_header) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("csv_to_matrix: cannot open file: " + path);
    }
    std::string line;
    std::size_t rows = 0;
    if (has_header) std::getline(file, line); // discard header
    while (std::getline(file, line)) {
        if (!line.empty()) ++rows;
    }
    return rows;
}

CSVResult csv_to_matrix(const std::string& path, bool has_header, bool normalize) {
    constexpr std::size_t kFeatures = 784;

    // Pass 1: count rows so we can allocate exactly once.
    const std::size_t N = count_data_rows(path, has_header);
    if (N == 0) {
        throw std::runtime_error("csv_to_matrix: no data rows found in " + path);
    }

    // Allocate outputs with final shapes.
    CSVResult out{ Matrix(kFeatures, N, 0.0f), std::vector<int>(N) };

    // Pass 2: parse and fill.
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("csv_to_matrix: cannot reopen file: " + path);
    }

    std::string line, cell;
    if (has_header) std::getline(file, line); // skip header

    std::size_t col = 0; // each CSV row becomes one column in X
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);

        // 1) Label
        if (!std::getline(ss, cell, ',')) {
            throw std::runtime_error("csv_to_matrix: missing label at row " + std::to_string(col));
        }
        try {
            out.y[col] = std::stoi(cell);
        } catch (...) {
            throw std::runtime_error("csv_to_matrix: non-integer label at row " + std::to_string(col));
        }

        // 2) Pixels
        std::size_t p = 0;
        while (p < kFeatures && std::getline(ss, cell, ',')) {
            float v;
            try {
                v = std::stof(cell);
            } catch (...) {
                throw std::runtime_error("csv_to_matrix: non-float pixel at row " + std::to_string(col));
            }
            if (normalize) v /= 255.0f;
            out.X(p, col) = v;   // feature = row p, example = column col
            ++p;
        }

        if (p != kFeatures) {
            throw std::runtime_error(
                "csv_to_matrix: expected 784 pixels, got " + std::to_string(p) +
                " at row " + std::to_string(col)
            );
        }

        ++col;
    }

    if (col != N) {
        throw std::runtime_error("csv_to_matrix: parsed rows (" + std::to_string(col) +
                                 ") != counted rows (" + std::to_string(N) + ")");
    }

    return out;
}

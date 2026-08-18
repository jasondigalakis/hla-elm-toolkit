// -----------------------------------------------------------------------------
// This file is part of hla_elm_toolkit, open-source software developed to
// accompany:
//   Kepentzis, S.; Chatzistamatiou, T.; Digalakis, J.; Petropoulou, O.;
//   Matsopoulos, G.K.; Koutsouris, D. "Development of an Extreme Learning
//   Machine approach to upgrade low/mid to high resolution HLA data
//   improving the usability of donor data from registries." Genes (MDPI).
//
// Software authors: Stavros Kepentzis (skepenjis@biomed.ntua.gr)
//                    Jason Digalakis  (jdigalakis@biomed.ntua.gr)
// Affiliation: Biomedical Engineering Laboratory, School of Electrical and
//              Computer Engineering, National Technical University of
//              Athens (NTUA), Athens, Greece
//
// Released as open-source software.
// -----------------------------------------------------------------------------

// Matrix.hpp
// Minimal, dependency-free dense matrix class supporting the linear
// algebra operations needed by the ELM family (multiply, transpose,
// add scaled identity, and solve a symmetric positive-(semi)definite
// linear system via Gaussian elimination with partial pivoting).
//
// This project deliberately avoids external dependencies (e.g. Eigen)
// so that it builds with nothing beyond a C++17 compiler and CMake.
// For registry-scale problems, swapping this class for a BLAS/LAPACK-
// or Eigen-backed implementation would be the natural next step; the
// public API here is small enough that this is a drop-in change.

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace hla_elm {

class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(std::size_t rows, std::size_t cols, double init = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, init) {}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    double& operator()(std::size_t r, std::size_t c) {
        return data_[r * cols_ + c];
    }
    double operator()(std::size_t r, std::size_t c) const {
        return data_[r * cols_ + c];
    }

    static Matrix identity(std::size_t n) {
        Matrix m(n, n, 0.0);
        for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
        return m;
    }

    // Random matrix with entries uniform in [lo, hi], for ELM's
    // randomly generated, untrained input-to-hidden weights (Section 2.4).
    static Matrix random_uniform(std::size_t rows, std::size_t cols,
                                  double lo, double hi, std::uint64_t seed) {
        Matrix m(rows, cols);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(lo, hi);
        for (auto& v : m.data_) v = dist(rng);
        return m;
    }

    Matrix transpose() const {
        Matrix out(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j)
                out(j, i) = (*this)(i, j);
        return out;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_)
            throw std::invalid_argument("Matrix::operator*: dimension mismatch");
        Matrix out(rows_, other.cols_, 0.0);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t k = 0; k < cols_; ++k) {
                double a = (*this)(i, k);
                if (a == 0.0) continue;
                for (std::size_t j = 0; j < other.cols_; ++j) {
                    out(i, j) += a * other(k, j);
                }
            }
        }
        return out;
    }

    Matrix operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::invalid_argument("Matrix::operator+: dimension mismatch");
        Matrix out(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] + other.data_[i];
        return out;
    }

    Matrix& add_scaled_identity(double scale) {
        if (rows_ != cols_) throw std::invalid_argument("add_scaled_identity: matrix must be square");
        for (std::size_t i = 0; i < rows_; ++i) (*this)(i, i) += scale;
        return *this;
    }

    // Elementwise sigmoid activation, matching the ELM hidden-layer
    // activation used throughout the article (Sections 2.4-2.6).
    Matrix sigmoid() const {
        Matrix out(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            double x = data_[i];
            if (x > 60) x = 60;
            if (x < -60) x = -60;
            out.data_[i] = 1.0 / (1.0 + std::exp(-x));
        }
        return out;
    }

    // Solve A * X = B for X, where A (this matrix) is square, via
    // Gaussian elimination with partial pivoting. Used to solve the
    // closed-form ELM output-weight equation
    // (H^T H + I/lambda) * beta = H^T T (Algorithm 1, line 18).
    Matrix solve(const Matrix& B) const {
        if (rows_ != cols_) throw std::invalid_argument("solve: matrix must be square");
        if (B.rows_ != rows_) throw std::invalid_argument("solve: rhs row mismatch");

        std::size_t n = rows_;
        std::size_t m = B.cols_;

        // Augmented matrix [A | B], row-major working copy.
        std::vector<double> aug(n * (n + m));
        auto A_at = [&](std::size_t r, std::size_t c) -> double& { return aug[r * (n + m) + c]; };
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) A_at(i, j) = (*this)(i, j);
            for (std::size_t j = 0; j < m; ++j) A_at(i, n + j) = B(i, j);
        }

        for (std::size_t col = 0; col < n; ++col) {
            // Partial pivot.
            std::size_t pivot = col;
            double best = std::fabs(A_at(col, col));
            for (std::size_t r = col + 1; r < n; ++r) {
                double v = std::fabs(A_at(r, col));
                if (v > best) { best = v; pivot = r; }
            }
            if (best < 1e-14) {
                // Nearly singular; nudge the diagonal (should not happen
                // in practice given the ridge term added before calling
                // solve(), but this keeps the routine robust).
                A_at(col, col) += 1e-10;
            } else if (pivot != col) {
                for (std::size_t j = 0; j < n + m; ++j)
                    std::swap(A_at(col, j), A_at(pivot, j));
            }

            double diag = A_at(col, col);
            for (std::size_t j = 0; j < n + m; ++j) A_at(col, j) /= diag;

            for (std::size_t r = 0; r < n; ++r) {
                if (r == col) continue;
                double factor = A_at(r, col);
                if (factor == 0.0) continue;
                for (std::size_t j = 0; j < n + m; ++j)
                    A_at(r, j) -= factor * A_at(col, j);
            }
        }

        Matrix X(n, m);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < m; ++j)
                X(i, j) = A_at(i, n + j);
        return X;
    }

    const std::vector<double>& raw() const { return data_; }

private:
    std::size_t rows_, cols_;
    std::vector<double> data_;
};

}  // namespace hla_elm

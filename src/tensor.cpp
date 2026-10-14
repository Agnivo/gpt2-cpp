// tensor.cpp — implementations for tensor.hpp. Definitions go here.
#include "tensor.hpp"

#include <assert.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <vector>

using namespace std;

namespace tn {

Tensor::Tensor(int rows, int cols) : rows_(rows), cols_(cols), row_stride_(cols), col_stride_(1) {
    data_ = vector<float>(rows * cols, 0.0f);
}

MatrixView Tensor::toMatrixView() const {
    return MatrixView(data_.data(), rows_, cols_, row_stride_, col_stride_);
}

void Tensor::set(int i, int j, float val) {
    assert(i >= 0 && i < rows_);
    assert(j >= 0 && j < cols_);
    int index = (i * row_stride_) + (j * col_stride_);
    assert(index >= 0 && index < static_cast<int>(data_.size()));
    data_[index] = val;
}

float Tensor::at(int i, int j) const {
    assert(i >= 0 && i < rows_);
    assert(j >= 0 && j < cols_);
    int index = (i * row_stride_) + (j * col_stride_);
    assert(index >= 0 && index < static_cast<int>(data_.size()));
    return data_[index];
}

int Tensor::rows() const { return rows_; }

int Tensor::cols() const { return cols_; }

float Tensor::val() const {
    assert(rows_ == 1 && cols_ == 1);
    return at(0, 0);
}

// Scale tensor and return the scaled tensor
Tensor Tensor::scale(float sf) const {
    Tensor out(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            out.set(i, j, at(i, j) * sf);
        }
    }
    return out;
}

float Tensor::sum() const {
    float sum = 0.0f;
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            sum += at(i, j);
        }
    }
    return sum;
}

int Tensor::argmax(int row) const {
    assert(row >= 0 && row < rows_);
    float max = -FLT_MAX;
    int max_idx = 0;
    for (int i = 0; i < cols_; i++) {
        if (at(row, i) > max) {
            max = at(row, i);
            max_idx = i;
        }
    }
    return max_idx;
}

Tensor Tensor::zeros(int rows, int cols) { return Tensor(rows, cols); }

Tensor Tensor::add_bias(const Tensor& bias) const {  // this: (rows, cols), bias: (1, cols)
    assert(bias.cols_ == cols_ && bias.rows_ == 1);
    Tensor output(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            output.set(i, j, at(i, j) + bias.at(0, j));
        }
    }
    return output;
}

Tensor Tensor::slice(int row_start, int row_end, int col_start, int col_end) {
    Tensor output = Tensor(row_end - row_start, col_end - col_start);
    for (int i = row_start; i < row_end; i++) {
        for (int j = col_start; j < col_end; j++) {
            output.set(i - row_start, j - col_start, at(i, j));
        }
    }
    return output;
}

// input: Tensor(batchsize, logits)
// output: Tensor(batchsize, probabilities)
Tensor Tensor::softmax() const {
    Tensor output = Tensor(rows_, cols_);

    // First find the max of the tensor values and then subtract the max from each
    // value so that the exp does not blow up
    // TODO: exp is being called twice. Cache it.
    for (int row = 0; row < rows_; row++) {
        float row_max = numeric_limits<float>::lowest();
        float row_sum = 0.0f;
        for (int col = 0; col < cols_; col++) {
            row_max = (at(row, col) > row_max ? at(row, col) : row_max);
        }
        for (int col = 0; col < cols_; col++) {
            row_sum += exp(at(row, col) - row_max);
        }
        for (int col = 0; col < cols_; col++) {
            output.set(row, col, exp(at(row, col) - row_max) / row_sum);
        }
    }
    return output;
}

// Tensor @ MatrixView -> Tensor
Tensor Tensor::matmul(const MatrixView& m) const { return tn::matmul(toMatrixView(), m); }

// Tensor @ Tensor -> Tensor
Tensor Tensor::matmul(const Tensor& t) const {
    return tn::matmul(toMatrixView(), t.toMatrixView());
}

Tensor Tensor::clone() const { return Tensor(rows_, cols_, row_stride_, col_stride_, data_); }

Tensor Tensor::transpose() const {
    Tensor t;
    t.data_ = data_;
    t.rows_ = cols_;
    t.cols_ = rows_;
    t.row_stride_ = col_stride_;
    t.col_stride_ = row_stride_;
    return t;
}

Tensor Tensor::operator/(const float b) const {
    Tensor out(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            out.set(i, j, at(i, j) * 1.0f / b);
        }
    }
    return out;
}

vector<size_t> Tensor::argsort() const {
    std::vector<size_t> idx(data_.size());
    // fill with 0, 1, ..., size() - 1
    std::iota(idx.begin(), idx.end(), 0);
    // sort by tensor.data[i]
    std::ranges::sort(idx, greater<>(), [&](size_t i) { return data_[i]; });

    return idx;
}

vector<float> Tensor::flatten() const {
    vector<float> output;
    output.reserve(rows_ * cols_);
    if (row_stride_ == cols_ && col_stride_ == 1)
        output = data_;
    else {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                output.push_back(at(i, j));
            }
        }
    }
    return output;
}

ostream& operator<<(ostream& os, const Tensor& t) {
    os << fixed << setprecision(5);
    for (int i = 0; i < t.rows(); ++i) {
        for (int j = 0; j < t.cols(); ++j) {
            os << setw(10) << t.at(i, j);
        }
        os << endl;
    }
    return os;
}

// MatrixView @ MatrixView -> Tensor
Tensor matmul(const MatrixView& a, const MatrixView& b) {
    assert(a.cols == b.rows);
    Tensor out(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float v = 0.0f;
            for (int k = 0; k < a.cols; k++) v += a.at(i, k) * b.at(k, j);
            out.set(i, j, v);
        }
    }
    return out;
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    assert(a.cols() == b.cols() && a.rows() == b.rows());
    Tensor out(a.rows(), a.cols());
    for (int i = 0; i < a.rows(); i++) {
        for (int j = 0; j < a.cols(); j++) {
            out.set(i, j, a.at(i, j) + b.at(i, j));
        }
    }
    return out;
}

}  // namespace tn

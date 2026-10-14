// tensor.hpp — only forward mode tensor library
#pragma once

#include <cassert>
#include <ostream>
#include <vector>

namespace tn {

struct MatrixView {
    const float* data;
    int rows, cols;
    int row_stride, col_stride;
    MatrixView() = default;
    MatrixView(const float* data, int rows, int cols)
        : data(data), rows(rows), cols(cols), row_stride(cols), col_stride(1) {}
    MatrixView(const float* data, int rows, int cols, int row_stride, int col_stride)
        : data(data), rows(rows), cols(cols), row_stride(row_stride), col_stride(col_stride) {}
    // self^T
    MatrixView transpose() const { return MatrixView(data, cols, rows, col_stride, row_stride); }
    float at(int i, int j) const {
        assert(i >= 0 && j >= 0 && i < rows && j < cols);
        return *(data + i * row_stride + j * col_stride);
    }
};

class Tensor {
   private:
    std::vector<float> data_;
    int rows_;
    int cols_;
    int row_stride_;
    int col_stride_;

   public:
    // Constructor for a tensor with a given fill value
    Tensor(int rows, int cols, float fill)
        : data_(rows * cols, fill), rows_(rows), cols_(cols), row_stride_(cols), col_stride_(1) {}

    // Constructor for a tensor with a given data vector
    Tensor(int rows, int cols, std::vector<float> data)
        : data_(std::move(data)), rows_(rows), cols_(cols), row_stride_(cols), col_stride_(1) {}

    // Constructor for a tensor with a given data vector and row/col strides
    Tensor(int rows, int cols, int row_stride, int col_stride, std::vector<float> data)
        : data_(std::move(data)),
          rows_(rows),
          cols_(cols),
          row_stride_(row_stride),
          col_stride_(col_stride) {}

    // Empty/undefined tensor (no storage). Assign into it later.
    Tensor() : rows_(0), cols_(0), row_stride_(0), col_stride_(0) {}

    // Allocate a rows x cols tensor, elements zero-initialized.
    Tensor(int rows, int cols);

    // Scale tensor and return the scaled tensor
    Tensor scale(float sf) const;

    static Tensor zeros(int rows, int cols);

    // misc helpers
    // convert a tensor to a MatrixView
    MatrixView toMatrixView() const;
    // Access to underlying raw storage
    float* raw() { return data_.data(); }
    // set element [i, j]
    void set(int i, int j, float val);
    // return element [i, j]
    float at(int i, int j) const;
    // number of rows and cols
    int rows() const;
    int cols() const;
    Tensor softmax() const;
    // find index of max element in given row of tensor
    int argmax(int row) const;
    // returns the value of a tensor. Makes sense only for a tensor with 1 value
    float val() const;
    // flatten a Tensor to a vector<float>
    std::vector<float> flatten() const;

    Tensor slice(int row_start, int row_end, int col_start, int col_end);

    // Tensor ops
    Tensor transpose() const;
    Tensor matmul(const Tensor&) const;
    Tensor matmul(const MatrixView& m) const;
    Tensor clone() const;
    float sum() const;                          // returns the sum of all elements of a tensor
    Tensor add_bias(const Tensor& bias) const;  // this: (rows, cols), bias: (1, cols)
    Tensor operator/(const float b) const;
    std::vector<size_t> argsort() const;
};

Tensor matmul(const MatrixView& a, const MatrixView& b);
Tensor operator+(const Tensor& a, const Tensor& b);

std::ostream& operator<<(std::ostream& os, const Tensor& t);

}  // namespace tn

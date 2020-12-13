// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef THEIA_MATH_MATRIX_RQ_DECOMPOSITION_H_
#define THEIA_MATH_MATRIX_RQ_DECOMPOSITION_H_

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

void kernel(float * A, float * Q, float * R);

// Mock kernel for testing
void mock_kernel(float * A, float * Q, float * R) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Q[j * 3 + i] = A[j * 3 + i] + 1;
      R[j * 3 + i] = A[j * 3 + i] + 2;
    }
  }
}

namespace theia {

// Implements the RQ decomposition that recovers matrices R and Q such that A =
// R * Q where R is an mxn upper-triangular matrix and Q is an nxn unitary
// matrix.
class RQDecomposition {
 public:
  // Shorthand for transpose type.
  typedef Eigen::Matrix<double,
                        3,
                        3,
                        Eigen::ColMajor> MatrixType;

  typedef Eigen::Matrix<double,
                        3,
                        3,
                        Eigen::ColMajor> MatrixTransposeType;

  typedef typename
  Eigen::HouseholderQR<MatrixTransposeType>::MatrixQType MatrixQType;

  // The matlab version of RQ decomposition is as follows:
  //
  // function [R Q] = rq(M)
  //   [Q,R] = qr(flipud(M)')
  //   R = flipud(R');
  //   R = fliplr(R);
  //   Q = Q';
  //   Q = flipud(Q);
  //
  // where flipup flips the matrix upside-down and fliplr flips the matrix from
  // left to right.
  explicit RQDecomposition(const MatrixType& matrix) {
    // flipud(M)' = fliplr(M').
    const MatrixTransposeType matrix_flipud_transpose =
        matrix.transpose().rowwise().reverse();

    // std::cout << "A:" << std::endl << matrix_flipud_transpose << std::endl;

    // Convert 3x3 input to standard C/C++ type
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> block_mat = matrix_flipud_transpose.cast<float>();
    float *A = block_mat.data();

    // Initialize outputs to zero
    float Q[9] = {};
    float R[9] = {};

    // Call our kernel!
    kernel(A, Q, R);

    // Convert standard C/C++ types back to eigen
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> q_matrix_float = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(Q, 3, 3);
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> r_matrix_float = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R, 3, 3);

    // Convert back to column major doubles
    Eigen::Matrix<double, 3, 3, Eigen::ColMajor> q_matrix = q_matrix_float.cast<double>();
    Eigen::Matrix<double, 3, 3, Eigen::ColMajor> r_matrix= r_matrix_float.cast<double>();

    // std::cout << "Q:" << std::endl << q_matrix << std::endl;
    // std::cout << "R:" << std::endl << r_matrix << std::endl;

    // Eigen call
    // Eigen::HouseholderQR<MatrixTransposeType> qr(matrix_flipud_transpose);


    const MatrixQType& q0 = q_matrix;
    const MatrixTransposeType& r0 = r_matrix;

    // Flip upside down.
    matrix_r_ = r0.transpose();
    matrix_r_ = matrix_r_.colwise().reverse().eval();

    // Flip left right.
    matrix_r_ = matrix_r_.rowwise().reverse().eval();

    // When R is an mxn matrix and m <= n then it is upper triangular. If m > n
    // then all elements below the subdiagonal are 0.
    for (int i = 0; i < matrix_r_.rows(); ++i) {
      for (int j = 0;
           matrix_r_.cols() - j > matrix_r_.rows() - i && j < matrix_r_.cols();
           ++j) {
        matrix_r_(i, j) = 0;
      }
    }

    // Flip upside down.
    matrix_q_ = q0.transpose();
    matrix_q_ = matrix_q_.colwise().reverse().eval();

    // Since the RQ decomposition is not unique, we will simply require that
    // det(Q) = 1. This makes using RQ decomposition for decomposing the
    // projection matrix very simple.
    if (matrix_q_.determinant() < 0) {
      matrix_q_.row(1) *= -1.0;
      matrix_r_.col(1) *= -1.0;
    }
  }

  // Mimic Eigen's QR interface.
  const MatrixType& matrixR() const {
    return matrix_r_;
  }

  const MatrixQType& matrixQ() const {
    return matrix_q_;
  }

 private:
  MatrixType matrix_r_;
  MatrixQType matrix_q_;
};

}  // namespace theia

#endif  // THEIA_MATH_MATRIX_RQ_DECOMPOSITION_H_

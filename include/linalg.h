#ifndef LINALG_H_
#define LINALG_H_

#include <cmath>
#include <utility>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace linalg {

// Storage layout
enum Layout { kRowMajor, kColMajor };

// Matrix template
template <typename T>
class Matrix {
    public:
        Matrix(size_t rows, size_t cols, Layout layout = kRowMajor);

        // Element access
        T& operator()(size_t i, size_t j);
        const T& operator()(size_t i, size_t j) const;

        Matrix<T> Transpose() const;
        T Trace() const;
        T FrobeniusNorm() const;
        size_t rows_;
        size_t cols_;

    private:
        Layout layout_;
        std::vector<T> data_;
};

// Vector is just an N×1 Matrix
template <typename T>
using Vector = Matrix<T>;

// Level-2 BLAS: y = α A x + β y
template <typename T>
void Gemv(const Matrix<T>& A,
          const Vector<T>& x,
          Vector<T>& y,
          T alpha = T(1),
          T beta  = T(0));

// Level-3 BLAS: C = α A B + β C
template <typename T>
void Gemm(const Matrix<T>& A,
          const Matrix<T>& B,
          Matrix<T>& C,
          T alpha = T(1),
          T beta  = T(0));

// Euclidean norm for Vector
template <typename T>
T Norm2(const Vector<T>& x);

// Jacobi rotation for symmetric eigenproblem
template <typename T>
std::pair<Matrix<T>, Vector<T>> JacobiEigen(const Matrix<T>& A,
                                            size_t max_iter = 100,
                                            T tol          = T(1e-10));

//Print matrix
template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& M);
}

#endif  // LINALG_H_

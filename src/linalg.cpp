#include "linalg.h"
#include <stdexcept>

namespace linalg {


    template <typename T>
    /**
     * @brief Constructor for Matrix class
     * @param rows Number of rows
     * @param cols Number of columns
     * @param layout Storage layout (row-major or column-major)
     * @return A Matrix object
     */
    Matrix<T>::Matrix(size_t rows, size_t cols, Layout layout)
        : rows_(rows),
        cols_(cols),
        layout_(layout),
        data_(rows * cols) {}

    template <typename T>
    /**
     * @brief Element access operator
     * @param i Row index
     * @param j Column index
     * @return Reference to the element at (i, j)
     */
    T& Matrix<T>::operator()(size_t i, size_t j) {
        if (layout_ == kRowMajor) {
            return data_[i * cols_ + j];
        } else {
            return data_[j * rows_ + i];
        }
    }

    template <typename T>
    /**
     * @brief Const element access operator
     * @param i Row index
     * @param j Column index
     * @return Const reference to the element at (i, j)
     */
    const T& Matrix<T>::operator()(size_t i, size_t j) const {
        if (layout_ == kRowMajor) {
            return data_[i * cols_ + j];
        } else {
            return data_[j * rows_ + i];
        }
    }

    template <typename T>
    /**
     * @brief Transpose the matrix
     * @return A new Matrix object that is the transpose of the original
     */
    Matrix<T> Matrix<T>::Transpose() const {
        Matrix<T> B(cols_, rows_, layout_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
            B(j, i) = (*this)(i, j);
            }
        }
        return B;
    }

    template <typename T>
    /**
     * @brief Compute the trace of the matrix
     * @return The sum of the diagonal elements
     */
    T Matrix<T>::Trace() const {
        if (rows_ != cols_) {
            throw std::runtime_error("Trace of non-square matrix");
        }
        T sum = T();
        for (size_t i = 0; i < rows_; ++i) {
            sum += (*this)(i, i);
        }
        return sum;
    }

    template <typename T>
    /**
     * @brief Compute the Frobenius norm of the matrix
     * @return The Frobenius norm
     */
    T Matrix<T>::FrobeniusNorm() const {
        T acc = T();
        for (const T& v : data_) {
            acc += v * v;
        }
        return std::sqrt(acc);
    }

    template <typename T>
    /**
     * @brief Level-2 BLAS: y = α A x + β y
     * @param A Matrix
     * @param x Vector
     * @param y Vector
     * @param alpha Scalar multiplier for A x
     * @param beta Scalar multiplier for y
     */
    void Gemv(const Matrix<T>& A, const Vector<T>& x, Vector<T>& y, T alpha, T beta) {
        if (A.cols_ != x.rows_ || A.rows_ != y.rows_) {
            throw std::runtime_error("Dimension mismatch in Gemv");
        }
        // y = β y
        for (size_t i = 0; i < y.rows_; ++i) {
            y(i, 0) = beta * y(i, 0);
        }
        // y += α A x
        for (size_t i = 0; i < A.rows_; ++i) {
            T tmp = T();
            for (size_t j = 0; j < A.cols_; ++j) {
            tmp += A(i, j) * x(j, 0);
            }
            y(i, 0) += alpha * tmp;
        }
    }

    template <typename T>
    /**
     * @brief Level-3 BLAS: C = α A B + β C
     * @param A Matrix
     * @param B Matrix
     * @param C Matrix
     * @param alpha Scalar multiplier for A B
     * @param beta Scalar multiplier for C
     */
    void Gemm(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, T alpha, T beta) {
        if (A.cols_ != B.rows_ ||
            A.rows_ != C.rows_ ||
            B.cols_ != C.cols_) {
            throw std::runtime_error("Dimension mismatch in Gemm");
        }
        // C = β C
        for (size_t i = 0; i < C.rows_; ++i) {
            for (size_t j = 0; j < C.cols_; ++j) {
            C(i, j) = beta * C(i, j);
            }
        }
        // C += α A B
        for (size_t i = 0; i < A.rows_; ++i) {
            for (size_t k = 0; k < A.cols_; ++k) {
            T a = A(i, k);
            for (size_t j = 0; j < B.cols_; ++j) {
                C(i, j) += alpha * a * B(k, j);
            }
            }
        }
    }

    template <typename T>
    /**
     * @brief Compute the Euclidean norm of a vector
     * @param x Vector
     * @return The Euclidean norm
     */
    T Norm2(const Vector<T>& x) {
        T acc = T();
        for (size_t i = 0; i < x.rows_; ++i) {
            acc += x(i, 0) * x(i, 0);
        }
        return std::sqrt(acc);
    }

    template <typename T>
    /**
     * @brief Jacobi eigenvalue algorithm for symmetric matrices
     * @param A Symmetric matrix
     * @param max_iter Maximum number of iterations
     * @param tol Tolerance for convergence
     * @return A pair of matrices: the eigenvectors and eigenvalues
     */
    std::pair<Matrix<T>, Vector<T>> JacobiEigen(const Matrix<T>& A, size_t max_iter, T tol) {
        if (A.rows_ != A.cols_) {
            throw std::runtime_error("JacobiEigen requires square matrix");
        }
        size_t n = A.rows_;
        Matrix<T> D = A;
        Matrix<T> V(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
            V(i, j) = (i == j ? T(1) : T(0));
            }
        }

        for (size_t iter = 0; iter < max_iter; ++iter) {
            size_t p = 0, q = 1;
            T max_off = std::abs(D(0, 1));
            // find largest off-diag
            for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                T val = std::abs(D(i, j));
                if (val > max_off) {
                max_off = val;
                p = i;
                q = j;
                }
            }
            }
            if (max_off < tol) break;
            // compute rotation
            T app = D(p, p), aqq = D(q, q), apq = D(p, q);
            T phi = (aqq - app) / (2 * apq);
            T t = (phi >= 0 ? 1 : -1) /
                (std::abs(phi) + std::sqrt(phi * phi + 1));
            T c = 1 / std::sqrt(t * t + 1);
            T s = t * c;
            // apply to D
            for (size_t i = 0; i < n; ++i) {
            if (i != p && i != q) {
                T dip = D(i, p), diq = D(i, q);
                D(i, p) = D(p, i) = c * dip - s * diq;
                D(i, q) = D(q, i) = c * diq + s * dip;
            }
            }
            D(p, p) = c*c*app - 2*c*s*apq + s*s*aqq;
            D(q, q) = s*s*app + 2*c*s*apq + c*c*aqq;
            D(p, q) = D(q, p) = T(0);
            // update V
            for (size_t i = 0; i < n; ++i) {
            T vip = V(i, p), viq = V(i, q);
            V(i, p) = c*vip - s*viq;
            V(i, q) = s*vip + c*viq;
            }
        }

        Vector<T> evals(n, 1);
        for (size_t i = 0; i < n; ++i) {
            evals(i, 0) = D(i, i);
        }
        return {V, evals};
    }

    template <typename T>
    /**
     * @brief Print the matrix
     * @param os Output stream
     * @param M Matrix
     * @return Output stream
     */
    std::ostream& operator<<(std::ostream& os, const Matrix<T>& M) {
        for (size_t i = 0; i < M.rows_; ++i) {
            os << "[ ";
            for (size_t j = 0; j < M.cols_; ++j) {
                os << M(i, j);
                if (j + 1 < M.cols_) os << ", ";
            }
            os << " ]";
            if (i + 1 < M.rows_) os << '\n';
        }
        return os;
    }

}

template class linalg::Matrix<double>;
template void linalg::Gemv<double>(
  const linalg::Matrix<double>&,
  const linalg::Vector<double>&,
  linalg::Vector<double>&,
  double,
  double
);
template void linalg::Gemm<double>(
  const linalg::Matrix<double>&,
  const linalg::Matrix<double>&,
  linalg::Matrix<double>&,
  double,
  double
);
template double linalg::Norm2<double>(const linalg::Vector<double>&);
template std::pair<linalg::Matrix<double>, linalg::Vector<double>>
linalg::JacobiEigen<double>(
  const linalg::Matrix<double>&,
  size_t,
  double
);
template std::ostream& linalg::operator<<(
  std::ostream&,
  const linalg::Matrix<double>&
);

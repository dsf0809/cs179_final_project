#include <iostream>
#include "linalg.h"

int main() {
  using namespace linalg;

  // 1) Matrix creation and element access
  Matrix<double> M(2, 2);
  M(0,0) = 1.0; M(0,1) = 2.0;
  M(1,0) = 3.0; M(1,1) = 4.0;
  std::cout << M << std::endl;

  // 2) Transpose
  auto Mt = M.Transpose();

  // 3) Trace
  std::cout << "Trace(M)        = " << M.Trace() << std::endl;

  // 4) Norms
  std::cout << "FrobeniusNorm  = " << M.FrobeniusNorm() << std::endl;
  Vector<double> v(2, 1);
  v(0,0) = 1.0; v(1,0) = 1.0;
  std::cout << "Vector Norm2   = " << Norm2(v) << std::endl;

  // 5) GEMV: y = M * v
  Vector<double> y(2, 1);
  Gemv(M, v, y);
  std::cout << "GEMV result    = \n" << y << std::endl;

  // 6) GEMM: C = M * I
  Matrix<double> I(2, 2);
  I(0,0) = 1.0; I(0,1) = 0.0;
  I(1,0) = 0.0; I(1,1) = 1.0;
  Matrix<double> C(2, 2);
  Gemm(M, I, C);
  std::cout << "GEMM result    = \n"<< C << std::endl;

  // 7) Jacobi eigen-decomposition of symmetric matrix
  Matrix<double> S(2, 2);
  S(0,0) = 2.0; S(0,1) = 1.0;
  S(1,0) = 1.0; S(1,1) = 2.0;
  auto [V, evals] = JacobiEigen(S);
  std::cout << "Eigenvalues    = \n" << evals << std::endl;

  return 0;
}

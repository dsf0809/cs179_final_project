#include "linalg.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace linalg;

  // Test 1: Trace and exception for non-square
  Matrix<double> I3(3,3);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      I3(i,j) = (i == j ? 1.0 : 0.0);
  assert(I3.Trace() == 3.0);

  Matrix<double> NS(2,3);
  bool caught = false;
  try {
    NS.Trace();
  } catch (const std::runtime_error&) {
    caught = true;
  }
  assert(caught);

  // Test 2: Frobenius norm
  Matrix<double> M2(2,2);
  M2(0,0) = 1; M2(0,1) = 2;
  M2(1,0) = 2; M2(1,1) = 1;
  double f = M2.FrobeniusNorm();
  assert(std::abs(f - std::sqrt(10.0)) < 1e-12);

  // Test 3: Norm2 zero and non-zero vectors
  Vector<double> z(3,1);
  for (size_t i = 0; i < 3; ++i) z(i,0) = 0.0;
  assert(Norm2(z) == 0.0);
  Vector<double> v2(2,1);
  v2(0,0) = 1.0; v2(1,0) = -2.0;
  assert(std::abs(Norm2(v2) - std::sqrt(5.0)) < 1e-12);

  // Test 4: GEMV with alpha & beta
  Matrix<double> A(2,2);
  A(0,0)=1; A(0,1)=2;
  A(1,0)=3; A(1,1)=4;
  Vector<double> x(2,1), y(2,1);
  x(0,0)=1; x(1,0)=1;
  y(0,0)=1; y(1,0)=1;
  // y = 2*A*x + 3*y_old => [9,17]
  Gemv(A, x, y, 2.0, 3.0);
  assert(y(0,0) == 9.0);
  assert(y(1,0) == 17.0);

  // Test 5: GEMM with alpha & beta
  Matrix<double> B(2,2), C(2,2);
  B(0,0)=0; B(0,1)=1;
  B(1,0)=2; B(1,1)=3;
  // C initially all ones
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      C(i,j) = 1.0;
  // C = A*B + C_old => [[5,8],[9,16]]
  Gemm(A, B, C, 1.0, 1.0);
  assert(C(0,0) == 5.0);
  assert(C(0,1) == 8.0);
  assert(C(1,0) == 9.0);
  assert(C(1,1) == 16.0);

  // Test 6: Transpose consistency
  Matrix<double> T(3,2);
  T(0,0)=1; T(0,1)=2;
  T(1,0)=3; T(1,1)=4;
  T(2,0)=5; T(2,1)=6;
  auto Tt = T.Transpose();
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 2; ++j)
      assert(T(i,j) == Tt(j,i));

  // Test 7: JacobiEigen sum of eigenvalues == trace
  Matrix<double> S3(3,3);
  S3(0,0)=2; S3(0,1)=1; S3(0,2)=0;
  S3(1,0)=1; S3(1,1)=2; S3(1,2)=1;
  S3(2,0)=0; S3(2,1)=1; S3(2,2)=2;
  auto [V3, E3] = JacobiEigen(S3);
  double sum_e = E3(0,0) + E3(1,0) + E3(2,0);
  assert(std::abs(sum_e - S3.Trace()) < 1e-6);

  std::cout << "All tests passed.\n";
  return 0;
}
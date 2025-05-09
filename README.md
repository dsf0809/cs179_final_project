# Linalg Library

A custom header-only C++ linear algebra library implementing basic BLAS-like routines, matrix operations, and Jacobi eigen-decomposition. Designed to replace external dependencies (e.g., Armadillo) in scientific codebases.


## Features

- **Templated Matrix\<T\> and Vector\<T\>:** dynamic size, row-/column-major storage.
- **Level-1 BLAS-like ops:** Euclidean norm (`Norm2`).
- **Level-2 BLAS-like ops:** matrix–vector multiply (`Gemv`).
- **Level-3 BLAS-like ops:** matrix–matrix multiply (`Gemm`).
- **Matrix utilities:** transpose (`Transpose`), trace (`Trace`), Frobenius norm (`FrobeniusNorm`).
- **Eigen-decomposition:** symmetric Jacobi rotation method (`JacobiEigen`).


## Folder Structure

```text
Final Project/
├── include/            # Public headers: definitions
│   └── linalg.h        
├── test/               # Unit tests
│   └── linalg_test.cpp 
├── demo/               # Demo programs
│   └── main.cpp        
├── src/                # Implementations
│   └── linalg.cpp      
├── CMakeLists.txt      # Build configuration
└── README.md           # This file
```


## Building

```bash
rm -rf build # ONLY WHEN if /build folder exist
mkdir build
cd build
cmake ..
make -j4
```


## Running the Demo

After building, from `build/`:

```bash
./linalg_demo    # runs examples/main.cpp
```

Expected output illustrates each function’s result (trace, norms, gemv, gemm, eigenvalues).


## Running Tests

Unit tests are in `test/linalg_test.cpp` using `assert()`. From `build/`:

```bash
./linalg_test
```

On success, you should see:

```
All tests passed.
```
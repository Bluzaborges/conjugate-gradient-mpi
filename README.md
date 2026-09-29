# Parallel Conjugate Gradient with MPI

A C implementation of the conjugate gradient method with MPI-based distributed matrix-vector multiplication. The matrix order is passed to the executable, and the process count is selected with `mpiexec`. Rank 0 prints its elapsed time and the iteration count.

The program solves the linear system `Ax = b`, where `A` is a symmetric positive-definite tridiagonal matrix with `4` on its main diagonal and `1` on the adjacent diagonals. The vector `b` is initialized with ones, and the initial solution vector `x` is initialized with zeros. Each process computes `order / process_count` rows of a matrix-vector product, and the partial results are combined with `MPI_Allgather`. **The matrix order must be a positive multiple of the process count**; other values are not supported by this implementation.

The convergence test uses the squared Euclidean norm of the residual, with a threshold of `1e-8` and a maximum of `100` iterations. Each process stores a dense copy of the matrix, requiring `O(n^2)` memory per process. The program does not validate numeric arguments or check allocation failures, so use a positive matrix order divisible by the process count and a size that fits in memory. This implementation is intended as an MPI study rather than a production distributed sparse linear algebra library.

## Requirements

- A C11-compatible compiler
- An MPI implementation such as Open MPI, MPICH, or Microsoft MPI
- CMake 3.16 or later (optional)

## Build

### Using an MPI compiler wrapper

```bash
mpicc -std=c11 -O2 -Wall -Wextra -pedantic src/gradient.c -o gradient
```

### Using CMake

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Usage

Choose the number of MPI processes with `-n` and pass a divisible matrix order to the program. After compiling with `mpicc`:

```bash
mpiexec -n 4 ./gradient 1000
```

After compiling with CMake using a single-configuration generator, use `./build/gradient` instead. With a multi-configuration generator, such as Visual Studio, use the executable under `build/Release/` (typically `gradient.exe` on Windows).

Example output (time and iteration count vary):

```text
Time: 0.0123 | Number of iterations: 8
```

Execution time varies by hardware, MPI implementation, compiler optimizations, matrix order, and process count.

## License

This project is licensed under the [MIT License](LICENSE).

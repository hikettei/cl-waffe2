
# cl-waffe2-simd

This package is separated from `cl-waffe2` by design, to avoid adding unnecessary dependencies for some users. (that is, this package does not necessarily need to be built). To enhance cl-waffe2 with fast SIMD-enabled operations, you MUST build this project.

https://github.com/rpav/cl-autowrap

## Supported Architectures

x86_64 ... avx2 avx512 sse2, AMD neon.

## TODO

Adding Sparse Matrix Support <- !!!

Sparse Gemm



# cl-waffe2-simd

This package is separated from `cl-waffe2` by design, to avoid adding unnecessary dependencies for some users. (that is, this package does not necessarily need to be built). To enhance cl-waffe2 with fast SIMD-enabled operations, you MUST build this package with `$ make build_simd_extension` at the top of the cl-waffe2 project tree.

## Supported Architectures

x86_64 ... avx2 (OK) avx512 sse2, AMD neon.

### Supported Dtypes

FP16 `double-float` `single-float` `uint32` `int32` `uint16` `int16` `uint8` `int8`

## TODO

Adding Sparse Matrix Support <- !!!

Sparse Gemm

Unfold/SIMD Mathematical Functions

OneDNN Supports

## Workflow

```lisp
[CPUTensor loads simd-extendion.dylib]
  | -> if not found -> reject-p and lisptensor
  v
execute ops
```


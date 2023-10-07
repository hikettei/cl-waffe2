
# [backend] :cl-waffe2/backends.cpu
The package `:cl-waffe2/backends.cpu` provides a `CPUTensor` backend which relies most of kernel implementations on foreign libraries invoked via CFFI. (e.g.: OpenBLAS, oneDNN in the coming future).
## Enabling the SIMD Extension

```sh
$ make build_simd_extension
```

See also: [cl-waffe2-simd](https://github.com/hikettei/cl-waffe2/tree/master/cl-waffe2-simd)

To get further performance on CPU, SIMD Extension must be installed on your device. This extension provides further SIMD-enabled CPUTensor operations (e.g.: !max/!min, Sparse Matrix Supports, vectorized mathematical functions of SLEEF, etc...). To use it, run `make build_simd_extension` in the same directory as cl-waffe2.asd. You can confirm that it works properly with the `(cl-waffe2:show-backends)` function.

## [AbstractTensor] CPUTensor

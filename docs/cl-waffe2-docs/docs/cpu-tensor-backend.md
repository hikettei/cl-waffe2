
# [package] :cl-waffe2/backends.cpu
The package `:cl-waffe2/backends.cpu` provides an AbstractTensor `CPUTensor` where most of its implementation relies on foreign libraries (e.g.: OpenBLAS, oneDNN in the coming future).
## Enabling the SIMD Extension

For some instructions (e.g.: `!max` `!min`, sparse matrix supports, `SLEEF`, etc...), packages that provide SIMD-enabled CPUTensor implementations are not enabled by default as a design. To enable it, run `make build_simd_extension` in the same directory as cl-waffe2.asd. You can check that it is loaded properly with the `(show-backends)` function.

## [AbstractTensor] CPUTensor

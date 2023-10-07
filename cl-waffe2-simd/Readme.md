
# cl-waffe2-simd

This package is separated from `cl-waffe2` by design, to avoid adding unnecessary dependencies for some users. (that is, this package does not necessarily need to be built). However, to enhance cl-waffe2 with fast SIMD-enabled operations on CPU, you **MUST** build this package with `$ make build_simd_extension` at the top of the cl-waffe2 project tree.

## Build

```sh
$ make                      ## from where GNUmakefile is available...
$ make build_simd_extension ## everything is ok with this command
```

## Supported Architectures

| Hardware       | State |
| -------------- | ----- | 
| SSE            | ❌    | 
| AVX(x86/64)    | ❌    | 
| AVX2(x86/64)   | ⭕️    |
| AVX512(x86/64) | ❌    |
| Neon(AMD)      | ❌    |

### Supported Dtypes

|Dtype            | State |
| --------------- | ----- |
| double-float    | ⭕️    |
| single-float    | ⭕️    |
| uint64          | ❌    |
| uint32          | ⭕️    |
| uint16          | ⭕️    |
| uint8           | ⭕️    |
| int64           | ❌    |
| int32           | ⭕️    |
| int16           | ⭕️    |
| int8            | ⭕️    |
| char            | ❌    |
| boolean(bit)    | ❌    |

## TODO

- Sparse Gemm

- Speed up `UnfoldNode` used in `Conv/Pool` with SIMD kernel

- Using some oneDNN APIs


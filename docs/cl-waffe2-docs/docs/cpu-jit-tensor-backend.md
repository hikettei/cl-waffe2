
# [package] :cl-waffe2/backends.jit.cpu
The package `:cl-waffe2/backends.jit.cpu` provides an AbstractTensor `JITCPUTensor` which accelerated by JIT Compiling to C code dynamically, (so this backend will require `gcc` as an additional requirement.)
## [parameter] `*default-c-compiler*`

Specify the command to compile the generated c codes. In default, "gcc".

## [parameter] `*compiler-flags*`

In default, `*compielr-flags*` = `'("-fPIC" "-O3" "-march=native")`

## [parameter] `*viz-compiled-code*`

Set t to display the compiled c code to terminal. In default, `nil`

## [AbstractTensor] JITCPUTensor
## [AbstractTensor] JITCPUScalarTensor

## [function] enable-cpu-jit-toplevel

```lisp
(enable-cpu-jit-toplevel (&key
			  (more-devices)
			  (compiler "gcc")
			  (viz-compiled-code nil)
                          (openmp nil)
			  (flags '("-fPIC" "-O3" "-march=native"))))
```

Sets `JITCPUTensor` and `JITCPUScalarTensor` to the top priority of backends. Place this function at the top of your code where JIT Compiling is needed. Of course, `JITCPUTensor` is developed as a one of `external backends` in cl-waffe2, therefore Local JIT compilation with the `with-devices` macro is another valid option.

### Inputs

`more-devices[List]` specify the list of device names. they have lower priority than `JITCPUTensor`

`viz-compiled-code[boolean]` Set t to display the compiled c codes.

`openMP[boolean]` set T to use OpenMP.

## [macro] with-cpu-jit

Under this macro, two backends (`JITCPUTensor` and `JITCPUScalarTensor`) are installed at the top of the priority list.

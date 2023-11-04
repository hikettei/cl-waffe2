
# [backend] :cl-waffe2/backends.jit.cpu

Those backends without JIT, relies on `do-compiled-loop` or `call-with-view` to calculate only part of matrices: complicated offsets and permution (i.e: `!view` and `!permute`). However, under certain circumstances this can be difficult to parallelise by simply calling a CFFI function; and this backend is intended to solve the problem.

This package provides a `JITCPUTensor` backend which works by jit-compiling whole code to vectorized C, and is only for the purpose of optimising memory layout, so currently only four arithmetic operations and copying are implemented. (OpFusion is remained to be implemented; but it is definitely possible to fuse several ops)

Optimise the memory layout by enclosing the code you want to optimise the layout in:

```lisp
(with-cpu-jit (CPUTensor LispTensor)
    ;; body
    )
```

Tips: Use the `proceed-bench` function to know the bottleneck; if `MoveTensorNode` combined with `PermuteNode` is slow compared to other nodes, the memory layout is remained to be optimized for example.

## [parameter] `*default-c-compiler*`

Specify the command to compile the generated c codes. In default, "gcc".

## [parameter] `*compiler-flags*`

In default, `*compielr-flags*` = `'("-fPIC" "-O3" "-march=native")`

## [parameter] `*viz-compiled-code*`

Set t to display the compiled c code to terminal. In default, `nil`

## [AbstractTensor] JITCPUTensor

```lisp
(with-devices (JITCPUTensor CPUTensor LispTensor)
    ;; Your code follows...
    )
```

## [function] cpujit-set-config

```lisp
(cpujit-set-config (&key
                      (compiler "gcc")
		      (viz-compiled-code nil)
                      (openmp nil)
	              (flags '("-fPIC" "-O3" "-march=native"))))
```

Declares configurations about JITCPUTensor. 
 
### Inputs

`compiler[string]` a compiler to use. in default set to gcc

`viz-compiled-code[boolean]` Set t to display generated C codes.

`openmp[boolean]` Set t to use OpenMP.

`flags[list]` additional compiler flags.

## [macro] with-cpu-jit

```lisp
(with-cpu-jit (&rest more-devices) &body body)
```

Under this macro, two backends (`JITCPUTensor` and `JITCPUScalarTensor`) are installed at the top of the priority list.

That is:

```lisp
`(with-devices (JITCPUTensor ,@more-devices)
     ,@body)
```

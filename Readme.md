
# cl-waffe2, Deep Learning Framework With Powerful Language, Common Lisp.

**The project is still in the concept stage.**

cl-waffe2 is a open-source project that provides a differentible matrix operation which is fairly extensible and strongly optimized by JIT Compiler.

Visit my preceding project: [cl-waffe](https://github.com/hikettei/cl-waffe).

# Features

### Safe and Static Shaping

### View First.

### Macro-Base JIT

### All Tensors and Nodes are Generic.

### Useful APIs

# Structure

Dependencies:
```lisp
cl-waffe2.asd (:serial = t)

[The Most Basic APIs (AbstractNode/AbstractTensor/JIT Compiler/Subscript Parser)]
1. ./source/vm/generic-tensor/
   ./source/vm/nodes/ (They're partially co-dependence)

[The Most Basic Nodes (AddNode/SubNode/Mathematical nodes etc...)]
2. ./source/base-impl/

[Implementations based on package (1.) (2.)]
3. ./source/backends/cpu
   ./source/backends/lisp
   ./source/nn
   ./source/optimizers
   ./source/viz
   etc...
```

# Workloads

- [x] Add Baseline: AbstractTensor
- [x] Multiple Backends
- [x] Pre-Inspection of Shapes
- [x] Fundamental APIs of view (Broadcasting, Slice, Indexing etc...)
- [x] Obvious and Intuitive message of Shape-Error.
- [x] A small DSL to write subscripts (i.e.: :where keyword)
- [x] A fundamental of forward/backward, and JIT. (acceptor)
- [x] Fundamental Dtypes
- [x] Displaying Tensor's all element in a small area.
- [x] Scheduling and Optimizing the allocation of array, by analyzing the computation node.
- [x] Pruning the rebundant computation node.
- [ ] Precompute the constant-node.
- [x] Basic Arithmetic Operation (+ - * /, and gemm)
- [ ] Support both column and row major ordering. (ignoring row-major gemm, it works)
- [ ] Sampling distributions (dense) -> Add More: gamma, chisquare distribution.
- [ ] Sampling distributions (sparse)
- [ ] Establish Specification About Dtype APIs, (e.g.: Casting from different dtypes, Auto Inferencing Dtype Of (make-tensor 1))
- [ ] Parallelize the computation node by lparallel.
- [ ] Let View APIs Support :indices :tflist (with cmp operations, bit-matrix)
- [ ] Eliminate Bugs
- [ ] swapaxes/transpose
- [ ] Add: !reshape with -1, unsqueeze, squeeze, repeat.
- [ ] CUDA Backend (Also, metal backends etc could be realised)
- [ ] Add Test Cases
- [ ] Give a strong features to cl-waffe2/viz
- [ ] Prepare documentations and examples
- [ ] Basic APIs for both LispTensor and CPUTensor.  (To Add: gemm without BLAS, impelement it as NoBlasMatmulTensor because it is signifcantly slow)
- [ ] Formulate specifications of nodes.
- [ ] Use Cl/CD
- [ ] ~~REPL-Friendly Node, (Implemented as proceed function)~~, with-dynamically-mode, set-config
- [ ] ascognitious
- [ ] node debugtools
- [x] Clarify runtime error, ~~backward error(OK)~~
- [ ] NN Features (Optimizers, etc...)
- [ ] Train MLP
- [ ] More clever Memory-management.
- [x] Mathematical and Dense Operations (exp log sin cos etc...)
- [ ] Operations like: argmax/argmin, reshape, transpose, swapaxes.
- [x] Optimize call-with-view, to minimize the number of using funcall. (i.e.: reshape (10 10) into (100) tensor)
- [x] Fix the issue where [~ a b] can't be applied to 2D Tensor.
- [ ] Optimized Sparse Matrix
- [ ] FP16 Matrix
- [ ] Add/Implement a SIMD Powered Backend for mathematical APIs. (named MathTensor), which provides (for example) approximation of exp in AVX512.
- [ ] (After released v1.0) cl-waffe2 for coalton.
- [ ] cl-waffe2/linalg, SVD
- [x] Distinguish the differences between Computed Tensor, and Not-Computed Tensor.
- [x] AOT Subscript-p
- [ ] optimize forward/backward
- [x] BugFix: !add x y <- x never resets. (the definition of sum contributed to this problem)
- [ ] optimize: incr vector ptr
- [ ] Fix: a ton of style warning
# Basics

### AbstractTensor

### What is build


# At first glance

```lisp
;; This form declares: General definition of MatMulNode.
(defnode (MatMulNode (myself &key transpose-a transpose-b)
	  :where (A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :slots ((transpose-a :initarg :transpose-a :type boolean :reader trans-a?)
		  (transpose-b :initarg :transpose-b :type boolean :reader trans-b?))
	  :documentation "(gemm 1.0 a b 0.0 c)"))

(define-impl (MatMulNode :device CPUTensor)
    ... ;; Give a implementation of MatmulNode when device=CPUTensor
    )

(define-impl (MatMulNode :device LispTensor)
    ... ;; Give a implementation of MatmulNode when device=LispTensor
   )
   
(defun build-kernel ()
  (with-devices (LispTensor)
    (let ((result (forward (MatmulNode) a b))) ;; This part works with Lisp
      (with-devices (CPUTensor)
        (let ((result1 (forward (MatmulNode) a b))) ;; This part works with BLAS
           (build (!sum result1)) ;; Applying JIT, waffe2 optimizes the computation node, memory-allocation, thread scheduling, etc...
	   )))))

(multiple-value-bind (fw bw vars params) (build-kernel)
    (time (funcall fw))
    (time (funcall bw)))
...
```

# Basic Workflow

```lisp
;; proceed is a little slow way (because it includes compile-time) but useful for debugging (define-by-run style)
(proceed (!add 1.0 1.0))

;; Fastest Way (define-and-run style)
(with-build (forward backward variables parameters) (!add 1.0 1.0)
    (funcall forward)
    (funcall backward)
    ...
    )
```


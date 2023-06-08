
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

# Workloads

- [x] AbstractTensor
- [x] Multiple Backends
- [x] Pre-Inspection of Shapes
- [x] Fundamental APIs of view (Broadcasting, Slice, Indexing etc...)
- [x] Obvious and Intuitive message of Shape-Error.
- [x] A small DSL to write subscripts (:where pharse)
- [x] A fundamental of forward/backward, and JIT. (acceptor)
- [x] Fundamental Dtypes
- [x] Displaying Tensor's all element in a small area.
- [x] Scheduling and Optimizing the allocation of array, by analyzing the computation node.
- [x] Pruning the rebundant computation node.
- [ ] Precompute the constant-node.
- [x] Basic Arithmetic Operation (+ - * /, and gemm)
- [ ] Support both column and row major ordering.
- [ ] Sampling distributions (dense)
- [ ] Sampling distributions (sparse)
- [ ] Establish Specification About Dtype APIs, (e.g.: Casting from different dtypes, Auto Inferencing Dtype Of (make-tensor 1))
- [ ] Parallelize the computation node by lparallel.
- [ ] Let View APIs Support :indices :tflist (with cmp operations, bit-matrix)
- [ ] Eliminate Bugs
- [ ] swapaxes/transpose
- [ ] CUDA Backend (Also, metal backends etc could be realised)
- [ ] Add Test Cases
- [ ] Give a strong features to cl-waffe2/viz
- [ ] Prepare documentations and examples
- [ ] Basic APIs for both LispTensor and CPUTensor.
- [ ] Formulate specifications of nodes.
- [ ] Use Cl/CD
- [ ] REPL-Friendly Node, (value tensor)
- [ ] ascognitious
- [ ] node debugtools
- [ ] Clarify runtime error.
- [ ] NN Features (Optimizers, etc...)
- [ ] Train MLP
- [ ] Optimize Memory-management.
- [ ] Mathematical and Dense Operations (exp log sin cos etc...)
- [ ] Operations like: argmax/argmin, reshape, transpose, swapaxes.
- [ ] Optimize call-with-view, to minimize the number of using funcall. (i.e.: reshape (10 10) into (100) tensor)
- [x] Fix the issue where [~ a b] can't be applied to 2D Tensor.
- [ ] Optimized Sparse Matrix
- [ ] FP16 Matrix

# Basics

### AbstractTensor

### What is build


# At first glance

```lisp
;; defines a node.
(defnode (MatMulNode (myself &key transpose-a transpose-b)
	  :where `(A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :slots ((transpose-a :initarg :transpose-a :type boolean :reader trans-a?)
		  (transpose-b :initarg :transpose-b :type boolean :reader trans-b?))
	  :documentation "(gemm 1.0 a b 0.0 c)"))

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


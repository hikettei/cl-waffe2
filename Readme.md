
# cl-waffe2, Deep Learning Framework With Powerful Language, Common Lisp.

**The project is still in the concept stage.**

cl-waffe2 is a open-source project that provides a differentible matrix operationwhich is fairly extensible and strongly optimized by JIT Compiler.

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
- [x] Pruning the computation node.
- [x] Basic Arithmetic Operation (+ - * /, and gemm)
- [ ] Support both column and row major ordering.
- [ ] Sampling distribution (dense)
- [ ] Sampling distribution (sparse)
- [ ] Casting from different dtypes.
- [ ] Parallelize the computation node by lparallel.
- [ ] Let View APIs Support :indices :tflist
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
- [ ] Optimize call-with-view, to minimize the number of using funcall.
- [ ] Fix the issue where [~ a b] can't be applied to 2D Tensor.



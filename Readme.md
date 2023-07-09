<p align="center">
    <a href="https://github.com/hikettei/cl-waffe2">
        <img alt="Logo" src="https://hikettei.github.io/cl-waffe-docs/cl-waffe.png" width="45%">
    </a>
    <br>
    <h3 align="center">Programmable Deep Learning Framework</h3>
    <p align="center">
    <a href="https://hikettei.github.io/cl-waffe2/"><strong>Visit the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/hikettei/cl-waffe2/issues">Issues</a>
    ·
    <a href="https://hikettei.github.io/cl-waffe2/install/">Installing</a>
    ·
    <a href="https://hikettei.github.io/cl-waffe2/overview/">Tutorials</a>
  </p>
</p>

# cl-waffe2

> ⚠️ cl-waffe2 is still in the experimental stage, things are subject to change. DO NOT USE CL-WAFFE2 IN YOUR PRODUCT.

cl-waffe2 provides a set of differentiable matrix operations which is aimed to apply to building a neural network model. Operations in cl-waffe2 are accelerated by `Lazy Evaluation` and `JIT Compiling with optimizing nodes.`.

Visit my preceding project: [cl-waffe](https://github.com/hikettei/cl-waffe).

# At first glance

(TODO: Example)

# References/Acknowledgments

https://www.jsoftware.com/papers/RationalizedAPL.htm

https://arxiv.org/pdf/1201.6035.pdf

https://www.european-lisp-symposium.org/static/2018/heisig.pdf

https://github.com/numpy/numpy/tree/main

https://pytorch.org/

https://github.com/melisgl/mgl-mat

https://dl.acm.org/doi/pdf/10.1145/359460.359482

https://andantesoft.hatenablog.com/entry/2023/04/30/183032

Marsaglia, G., & Tsang, W. W. (2000). The ziggurat method for generating random variables. Journal of statistical software.

https://marui.hatenablog.com/entry/2023/01/23/194507

https://arxiv.org/abs/1912.01703

(More to be added...)



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
- [x] ~~Precompute the constant-node.~~ (Partially done?)
- [x] Basic Arithmetic Operation (+ - * /, and gemm)
- [ ] Support: Row/Column Major Tensor, and `gemm` (currently tests is too few.)
- [x] ~~Sampling distributions (dense) -> Add More: gamma, chisquare distribution.~~ -> Add: `Orthogonal`
- [x] Sampling distributions (sparse)
- [ ] Features on casting dtype is still not enough.
- [ ] Parallelize the computation node by lparallel. (No need to do this?)
- [ ] Let View APIs Support :indices :tflist (with cmp operations, bit-matrix)
- [ ] View API: permute first iteration
- [x] Add: ~~!reshape with -1, unsqueeze, squeeze, repeat.~~ -> repeat/unsqueeze/squeeze is remained to be implemented.
- [ ] CUDA Backend (Also, metal backends etc could be realised).
- [ ] Give a strong features to cl-waffe2/viz
- [x] Prepare documentations and examples (ongoing)
- [x] Basic APIs for both LispTensor and CPUTensor.  (To Add: gemm without BLAS, impelement it as NoBlasMatmulTensor because it is signifcantly slow)
- [x] Formulate specifications of nodes.
- [ ] Use Cl/CD
- [x] ~~REPL-Friendly Node, (Implemented as proceed function)~~, ~~with-dynamically-mode (no need to do this)~~, set-config
- [ ] ascognitious
- [ ] node debugtools
- [x] Clarify runtime error, ~~backward error(OK)~~
- [ ] NN Features (Optimizers, etc...)
- [ ] Train MLP
- [x] ~~More clever Memory-management.~~ -> Added memory-pool, but theres a lot of room to be improved.
- [x] Mathematical and Dense Operations (exp log sin cos etc...)
- [x] Operations like: argmax/argmin, reshape, transpose, swapaxes.
- [x] Optimize call-with-view, to minimize the number of using funcall. (i.e.: reshape (10 10) into (100) tensor)
- [x] Fix the issue where [~ a b] can't be applied to 2D Tensor.
- [ ] Optimized Sparse Matrix
- [ ] FP16 Matrix
- [ ] Add/Implement a SIMD Powered Backend for mathematical APIs. (named MathTensor), which provides (for example) approximation of exp in AVX512. It is not portable but written in C/C++ can called via cffi. (use SLEEF?)
- [ ] (After released v1.0) cl-waffe2 for coalton.
- [ ] cl-waffe2/linalg, SVD
- [x] Distinguish the differences between Computed Tensor, and Not-Computed Tensor.
- [x] AOT Subscript-p
- [x] optimize forward/backward
- [x] BugFix: !add x y <- x never resets. (the definition of sum contributed to this problem)
- [ ] Fix: a ton of style warning
- [ ] lparallel -> optimized-memory-allocation -> fast-math kernel, fp8 fp16, uint4 etc...
- [ ] Add: Restarting

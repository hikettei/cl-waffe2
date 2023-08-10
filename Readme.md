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

[![CI](https://github.com/hikettei/cl-waffe2/actions/workflows/Cl.yml/badge.svg?branch=master)](https://github.com/hikettei/cl-waffe2/actions/workflows/Cl.yml)

# cl-waffe2

> ⚠️ cl-waffe2 is still in the experimental stage, things are subject to change. DO NOT USE CL-WAFFE2 IN YOUR PRODUCT.

cl-waffe2 is an experimental Deep Learning Framework working on Common Lisp which dynamically compiles Optimal Common Lisp codes generated from the portable and user-extensible APIs in runtime.
As a future goal, I'm working on JIT compilation to C++/CUDA, and providing a framework dedicated to solving optimizing/inferencing deep neural network models using AD, all in Common Lisp. This framework is decoupled from other existing libraries by design, but interoperating with other libraries by efficient use of other libraries via with-facet macros and Common Lisp standard arrays is strongly recommended. I'm just focusing on efficient AD.

Every operation in cl-waffe2 is lazy evaluated and later compiled, and there are two valid functions to compile/execute nodes: `proceed` and `build`. With the `proceed` function, cl-waffe2 works as if it is an interpreter, with no compiling overhead in runtime, and it is differentiable.  On the other hand the function `build` will generate codes which are fully optimized for training models.

Portability to other devices is a major concern. In particular, cl-waffe2 is designed to put as few barriers between the user and the developer as possible.

Visit my preceding project: [cl-waffe](https://github.com/hikettei/cl-waffe).

### Workloads

- [x] Establish a baseline for generic matrix operations from zero for Common Lisp, as practised by Julia.
    - Four data structures used in deep learning: `AbstractNode` `AbstractTensor` `AbstractOptimizer`, and `AbstractTrainer`.
    - Fundamental APIs: `View API/Broadcasting/Permution` `NDArrays with multidimensonal offsets` `build/proceed` `facet of tensors`
    - Graph-level optimization of computation nodes. (pruning unused nodes/in-place mutation/inlining view computations)
    - For experiments, implement a backend that runs at minimum speed: `LispTensor`.

- [ ] Construct JIT Compiler from cl-waffe2 to `C`. and fuse operations and loops. + Multi-Threading (work in progress)

- [ ] Add basic computation nodes for controling nodes: `MapNode` `IfNode` etc...

# Concepts

## Frontend and Backend Separation

In the design phase, cl-waffe2 separates the `cl-waffe2/base-impl` package, which provides an abstract definition of operations (i.e.: the definition of `AbstractNode` by using the `defnode` macro), from each `cl-waffe2/backends` package, which gives its implementation (e.g.: `define-impl` macro). Your programme builds the network by dynamically referring to the `*using-device*` parameter while deciding which implementation of `AbstractTensor` to use. This allows users to implement existing backend (or create a new one) re-implementations and extensions of instructions without any constraints.

For example, an AbstractTensor that works with Common Lisp standard arrays can be created as follows:

```lisp
;; MyTensor extends CPUTensor extends AbstractTensor
(defclass MyTensor (CPUTensor) nil)
```

The `with-devices` macro is used to declare the devices to be used along with their priority, computation nodes build based on it.

```lisp
(with-devices (MyTensor CPUTensor)
    ;; Under this scope, Priority = (MyTensor -> CPUTensor)
    (!add (randn `(3 3)) (randn `(3 3))))

{MYTENSOR[float] :shape (3 3) :named ChainTMP12737 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (3 3) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>}

;; Proceed is a differentiable operation which compiles/evaluates previous computation nodes.
(proceed *) ;; MyTensor has no any implementation for AddNode, so CPUTensor is returned.

{CPUTENSOR[float] :shape (3 3) :named ChainTMP12759 
  :vec-state [computed]
  ((-0.9171257  0.4143868   0.9511917)
   (2.224929    1.4860398   0.8402364)
   (0.051592022 0.5673465   -0.46694738))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

This helps Common Lisp, a dynamically typed language, to use valuable information such as matrix rank and shape in advance, to detect shape errors before performing operations, or to generate optimal codes.

See [this section](https://hikettei.github.io/cl-waffe2/base-impl-nodes/) for the specifications under which computation nodes are defined.

## Lazy-evaluation and Extensible JIT Compiler

cl-waffe2 is a lazy-evaluation first framework in which each computation node is represented by `AbstractNode` and `S-expression` with the strong restrict of `A <- Op(B, C, ...)`. That is, cl-waffe2 specializes on computing DAGs in a efficient way (e.g.: in-place mutation, computing offsets in advance, no runtime allocation, and more...). In addition to that, it allows the creation of compilers to external languages by extending the device, making it easy to create compilers to C (provides as `JITCPUTensor` in standard) and CUDA.

See also:

- https://hikettei.github.io/cl-waffe2/overview/#in-place-optimizing
- https://hikettei.github.io/cl-waffe2/vm/
- https://hikettei.github.io/cl-waffe2/cpu-jit-tensor-backend/

## Powerful Network Description Features

TODO: `defsequence/call->` `composite/node` `defmodel`

`defmodel` defines a set of nodes.

```lisp
(defmodel (Softmax-Model (self)
       :where (X[~] -> OUT[~])
       :on-call-> ((self x)
               (declare (ignore self))
               (let* ((x1 (!sub x (!mean x  :axis 1 :keepdims t)))
                      (z  (!sum   (!exp x1) :axis 1 :keepdims t)))
                  (!div (!exp x1) z)))))
```

It can be used to lazily evaluate and compile later, or to define functions for immediate execution.

`call` to keep using lazy-evaluation.

```lisp
(call (Softmax-Model) (randn `(10 10)))

{CPUTENSOR[float] :shape (10 10) :named ChainTMP13029 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (10 10) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: DIVNODE-LISPTENSOR (A[~] B[~] -> A[~])>}

(proceed *) ;; Being Compiler Later, by proceed or build
{CPUTENSOR[float] :shape (10 10) :named ChainTMP13158 
  :vec-state [computed]
  ((0.05213483   0.11118897   0.107058994  ~ 0.1897892    0.055277593  0.028915826)                    
   (0.0025042535 0.3952663    0.0109358365 ~ 0.033085804  0.04627693   0.14064543)   
                 ...
   (0.067338936  0.06604112   0.065211095  ~ 0.051910892  0.10963429   0.060249455)
   (0.029982507  0.31893584   0.18214627   ~ 0.015864253  0.2993634    0.02982553))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

`define-composite-function` to define a function.

```lisp
(define-composite-function (Softmax-Model) !softmax-static)

(time (!softmax-static (randn `(10 10)))) ;; No compiling time for second and subsequent calls.
Evaluation took:
  0.000 seconds of real time
  0.000460 seconds of total run time (0.000418 user, 0.000042 system)
  100.00% CPU
  1,073,976 processor cycles
  32,512 bytes consed
  
{CPUTENSOR[float] :shape (10 10) :named ChainTMP13578 
  ((0.06095955   0.06010023   0.03573166   ~ 0.01910117   0.036269512  0.03422032)                    
   (0.3116705    0.041012052  0.012784039  ~ 0.08029219   0.062023237  0.03468513)   
                 ...
   (0.057693116  0.19069833   0.061993677  ~ 0.20243406   0.02019287   0.07737376)
   (0.35623857   0.038911298  0.028082697  ~ 0.050502267  0.024571734  0.10532298))
  :facet :input
  :requires-grad NIL
  :backward NIL}
```

There's more, `defnode` is a generic definiiton of `AbstractNode`, being implemented by `define-impl` which works like a macro in Common Lisp. On the other hand, `define-static-node` works like a `defun`. For details, visit docs: https://hikettei.github.io/cl-waffe2/overview/#network-units-node-and-composite.

## Numpy-like APIs

Except that you need to call `proceed` or `build` at the end of the operation, cl-waffe2 APIs was made to be similar to Numpy. In addition, cl-waffe2 is intended to work with REPL. (ease of debugging needs to be improved though...)

See also: https://hikettei.github.io/cl-waffe2/base-impl/

## All in the simple APIs

The combination of delay evaluation and node definition mechanisms allows all the shapes of the network to be specified without the need to write special code.

```lisp
(defsequence MLP-Sequence (in-features hidden-dim out-features
               &key (activation #'!tanh))
         "3 Layers MLP"
         (LinearLayer in-features hidden-dim)
         (asnode activation)
         (LinearLayer hidden-dim hidden-dim)
         (asnode activation)
         (LinearLayer hidden-dim out-features)
         (asnode #'!softmax))
```

```lisp
(MLP-Sequence 784 512 256)

<Composite: MLP-SEQUENCE{W23852}(
    <<6 Layers Sequence>>

[1/6]          ↓ 
<Composite: LINEARLAYER{W23682}(
    <Input : ((~ BATCH-SIZE 784)) -> Output: ((~ BATCH-SIZE 512))>

    WEIGHTS -> (512 784)
    BIAS    -> (512)
)>
[2/6]          ↓ 
<Composite: ENCAPSULATED-NODE{W23680}(
    #<FUNCTION !TANH>
)>
[3/6]          ↓ 
<Composite: LINEARLAYER{W23510}(
    <Input : ((~ BATCH-SIZE 512)) -> Output: ((~ BATCH-SIZE 512))>

    WEIGHTS -> (512 512)
    BIAS    -> (512)
)>
[4/6]          ↓ 
<Composite: ENCAPSULATED-NODE{W23508}(
    #<FUNCTION !TANH>
)>
[5/6]          ↓ 
<Composite: LINEARLAYER{W23338}(
    <Input : ((~ BATCH-SIZE 512)) -> Output: ((~ BATCH-SIZE 256))>

    WEIGHTS -> (256 512)
    BIAS    -> (256)
)>
[6/6]          ↓ 
<Composite: ENCAPSULATED-NODE{W23336}(
    #<FUNCTION CL-WAFFE2/NN:!SOFTMAX>
)>)>
```

# Experiments

(TODO)
Training time/accuracy with MNIST/Cifar-10 by MLP/CNN compared to Keras/Tensorflow/PyTorch.

# References/Acknowledgments


・All comments on this Reddit post: [Does anyone have any interest in my deep-learning framework?](https://www.reddit.com/r/Common_Lisp/comments/124da1l/does_anyone_have_any_interest_in_my_deeplearning/).

- digikar99 for giving me intriguing perspectives on some semantics and the publication of a large number of valuable references.
    - [Features of Common Lispy approach to (Better) Numpy](https://gist.github.com/digikar99/ba2f0bb34021bfdc086b9c1c712ca228)

- [Rationalized APL](https://www.jsoftware.com/papers/RationalizedAPL.htm)

- [Previous research of Petalisp](https://www.european-lisp-symposium.org/static/2018/heisig.pdf)
    - https://github.com/marcoheisig/Petalisp/tree/master

- [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767)

- Some of the algorithms implemented within the source code are referenced below:

    - https://arxiv.org/pdf/1201.6035.pdf

    - https://github.com/melisgl/mgl-mat

    - https://dl.acm.org/doi/pdf/10.1145/359460.359482

    - https://andantesoft.hatenablog.com/entry/2023/04/30/183032

    - Marsaglia, G., & Tsang, W. W. (2000). The ziggurat method for generating random variables. Journal of statistical software.

    - https://marui.hatenablog.com/entry/2023/01/23/194507

    - https://arxiv.org/abs/1912.01703

- Previous works of JIT Compiler for Deep Learning:

    - https://arxiv.org/pdf/2002.03794.pdf

    - See also my reading list: https://github.com/hikettei/cl-waffe2/issues/47

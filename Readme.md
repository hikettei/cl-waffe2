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
    <a href="https://github.com/hikettei/cl-waffe2/tree/master/examples">Examples</a>
  </p>
</p>

[![CI](https://github.com/hikettei/cl-waffe2/actions/workflows/Cl.yml/badge.svg?branch=master)](https://github.com/hikettei/cl-waffe2/actions/workflows/Cl.yml)

# cl-waffe2

> ⚠️ cl-waffe2 is still in the experimental stage. Things are subject to change, and APIs can be changed without warnings. DO NOT USE CL-WAFFE2 IN YOUR PRODUCT.

cl-waffe2 provides fast, systematic, easy to optimize, customizable, and environment- and device- independent abstract matrix operations and reverse mode tape-based Automatic Differentiation on Common Lisp. Plus, we also provide features for building and training neural network models, powered by `JIT Compiler`.

Visit my preceding project (not relevant to the cl-waffe2 project): [cl-waffe](https://github.com/hikettei/cl-waffe).

## ✨Features

- cl-waffe2 brings `AbstractTensor` to Common Lisp.
- Extensible: Operations can be extended/reimplemented with any matrix operation libraries you like! Plus, No code rewriting when changing devices.
- Inlining:  Anyone can write a optimized loop iteration, for example, `Loop Collapse` and `Loop Fusion`, computing offsets in advance, and scheduling multi-threading by `lparallel`.
- Profiling: Super easy to know the bottleneck in your network, because a `proceed-bench` function profiles every instruction.
- Nodes: Systematic macros for building computation nodes and easy to visualize!
- 👏 Its core part and VM are 100% written on **ANSI Common Lisp**.

## 🍃 Quicklook

In the simplest example, the `build` function traces and compiles the network from the endpoints of the computation nodes. 

```lisp
(let ((a (make-input `(A B) :A))
      (b (make-input `(A B) :B)))
  (let ((model (build (!sum (!mul a b)) :inputs `(:A :B))))
    (print model)
    ;; model is a compiled function: f(a b)
    (forward model (randn `(3 3)) (randn `(3 3)))))

;;<Compiled-Composite
;;    forward  : forward(model A B) -> CPUTENSOR{FLOAT}(1 1)
;;    backward : backward(model) -> t
;;    inputs:
;;        A -> (A B)
;;        B -> (A B)
;;>

;;{CPUTENSOR[float] :shape (1 1) -> :view (<(BROADCAST 1)> <(BROADCAST 1)>) -> :visible-shape (1 1) :named ChainTMP646587 
;;  ((1.0858848))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward NIL} 
```

# 📈 Experiments

(TODO)

# 📕 References/Acknowledgments

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

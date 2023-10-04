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
    <a href="https://github.com/hikettei/cl-waffe2/blob/master/docs/cl-waffe2-docs/docs/overview.md">Concepts</a>
    ·
    <a href="https://hikettei.github.io/cl-waffe2/install/">Install</a>
    ·
    <a href="https://github.com/hikettei/cl-waffe2/tree/master/examples">Examples</a>
  </p>
</p>

[![CI](https://github.com/hikettei/cl-waffe2/actions/workflows/Cl.yml/badge.svg?branch=master)](https://github.com/hikettei/cl-waffe2/actions/workflows/Cl.yml)

# cl-waffe2

> ⚠️ cl-waffe2 is still in the experimental stage. Things are subject to change, and APIs can be changed without warnings. DO NOT USE CL-WAFFE2 IN YOUR PRODUCT.
> 
> I actually have a repository [cl-waffe(DEPRECATED UNSUPPORTED!)](https://github.com/hikettei/cl-waffe) with a similar name. Don't misunderstand that: cl-waffe**2** is the latest one and all features are inherited from the old one.

cl-waffe2 provides fast, systematic, easy to optimize, customizable, and environment- and device- independent abstract matrix operations, and reverse mode tape-based Automatic Differentiation on Common Lisp. Plus, we also provide features for building and training neural network models, accelerated by JIT Compiler.

Roughly speaking, this is a framework for the graph and tensor abstraction without overheads. All features provided here can be extended by users without exception. And with the minimal code. cl-waffe2 is designed as the truly easiest framework to write extensions by users. There's no barrier between users and developers. There's no restriction imposed by framework ignoring Common Lisp.

Its abstraction layers are almost reaching the goals and enough practical, but there is still a serious lack of backend functionality, and documentations. Contributions are welcome and feel free to contact me: [hikettei](https://github.com/hikettei) if you've interested in this project.

## ✨Features

- cl-waffe2 brings **AbstractTensor** to Common Lisp.
- **Extensible** All operations can be reimplemented with any matrix operation libraries you like! Plus, AbstractNode guarantees that no code rewriting is needed when changing devices.
- **Inlining**  Anyone can write an optimized loop calling foreign libraries; an order is collapsed and shuffled depending on the ranks and offsets. 
- **Graph-Level Optimization** cl-waffe2 provides a powerful abstract graph optimization tool that can be used on any devices. For example, it optimizes the locality of memory, and make operations in-place as much as possible.
- **Visualize** Super easy to know the bottleneck in your network, because a `proceed-bench` function profiles every instruction.
- **Debugging** cl-waffe2 is enough clever that not only detecting all Shaping-Error before the execution but also suggests alternatives! In addition, All objects in cl-waffe2 are nicely rendered on your REPL.
- **Systematic Nodes**: AbstractNodes and Models are written with small codes. Moreover, they're easy to compose and compile.
- **Symbolic Differentiation** In the first place, cl-waffe2 do not create nodes that are later modified. Compiler macros eliminate functions producing such nodes.

## 🍃 Quicklook

In the simplest example, the `build` function traces and compiles the network from the endpoints of the computation nodes. 

```lisp
(let ((a (make-input `(A B) :A))
      (b (make-input `(A B) :B)))
  (let ((model (build (!sum (!mul a b)) :inputs `(:A :B))))
    (print model)
    ;; model is a compiled function: f(a b)
    (forward model (randn `(3 3)) (randn `(3 3)))))

;;<Compiled-Composite(allocated-p=NIL)
;;    forward     : forward(model A B) -> CPUTENSOR{FLOAT}(1 1)
;;    backward    : backward(model) -> t
;;    memory-pool : two tensor(s)
;;                   L {8.0e-6+((A B) x 4.0e-6)}MB
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

We also provide [example projects](https://github.com/hikettei/cl-waffe2/tree/master/examples) here!

# 📈 Experiments

(TODO)

# 📕 References and Acknowledgments

- All comments on this Reddit post: [Does anyone have any interest in my deep-learning framework?](https://www.reddit.com/r/Common_Lisp/comments/124da1l/does_anyone_have_any_interest_in_my_deeplearning/).

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

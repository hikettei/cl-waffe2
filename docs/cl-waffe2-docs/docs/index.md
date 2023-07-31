
<p align="center">
    <a href="https://github.com/hikettei/cl-waffe2">
        <img alt="Logo" src="https://hikettei.github.io/cl-waffe-docs/cl-waffe.png" width="45%">
    </a>
    <br>
    <h3 align="center">Programmable Deep Learning Framework</h3>
    <p align="center">
    <a href="https://github.com/hikettei/cl-waffe2"><strong>Repository »</strong></a>
    <br />
    <br />
    <a href="https://github.com/hikettei/cl-waffe2/issues">Issues</a>
    ·
    <a href="./install">Installing</a>
    ·
    <a href="./overview">Tutorials</a>
  </p>
</p>

> __⚠️ cl-waffe2 is still in the concept stage and has a limited feature. Also, things are subject to change. DO NOT USE CL-WAFFE2 IN YOUR PRODUCT.__

cl-waffe2 is an experimental Deep Learning Framework working on Common Lisp which dynamically compiles Optimal Common Lisp codes generated from the portable and user-extensible APIs in runtime.
As a future goal, I'm working on JIT compilation to C++/CUDA, and providing a framework dedicated to solving optimizing/inferencing deep neural network models using AD, all in Common Lisp. This framework is decoupled from other existing libraries by design, but interoperating with other libraries by efficient use of other libraries via with-facet macros and Common Lisp standard arrays is strongly recommended. I'm just focusing on efficient AD.

Every operation in cl-waffe2 is lazy evaluated and later compiled, and there are two key functions to compile and execute nodes: `proceed` and `build`. With the `proceed` function, cl-waffe2 works as if it is an interpreter, with no compiling overhead in runtime, and it is differentiable.  On the other hand the function `build` will generate codes which are fully optimized for training models.

Portability to other devices is a major concern. In particular, cl-waffe2 is designed to put as few barriers between the user and the developer as possible.

### Workloads

- [x] Establish a baseline for generic matrix operations from zero for Common Lisp, as practised by Julia.
    - Four data structures used in deep learning: `AbstractNode` `AbstractTensor` `AbstractOptimizer`, and `AbstractTrainer`.
    - Fundamental APIs: `View API/Broadcasting/Permution` `NDArrays with multidimensonal offsets` `build/proceed` `facet of tensors`
    - Graph-level optimization of computation nodes. (pruning unused nodes/in-place mutation/inlining view computations)
    - For experiments, implement a backend that runs at minimum speed: `LispTensor`.

- [ ] Construct JIT Compiler from cl-waffe2 to `C++`. and fuse operations and loops.

- [ ] Add basic computation nodes for controling nodes: `MapNode` `IfNode` etc...

This project has been developed on these primal concepts:

1. `Do not run until the node is optimized.`, that is, Lazy-Evaluation first.

2. Generic operations powered by multiple kinds of user-defined backends.

3. defined-and-run, closer to defined-by-run APIs.


**Everything in this documentation is still incomplete! I guess should be much more polished!**

#

## What's next?

1. [Install cl-waffe2](./install)

2. [Learn the concepts with practical tutorials](./overview)

3. [Visit some examples! (Not yet ready...)](./examples)

## Benchmarks

```
Coming soon ...
```

## Contribute

```
Coming soon ...
```

## Acknowledgments

```
Coming soon...
```

## LICENCE

```
Coming soon...
```
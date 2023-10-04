#

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

[![CI](https://github.com/hikettei/cl-waffe2/actions/workflows/Cl.yml/badge.svg?branch=master)](https://github.com/hikettei/cl-waffe2/actions/workflows/Cl.yml)

## Introduction

> ⚠️ cl-waffe2 is still in the experimental stage. Things are subject to change, and APIs can be changed without warnings. DO NOT USE CL-WAFFE2 IN YOUR PRODUCT.
> 
> I actually have a repository [cl-waffe(DEPRECATED UNSUPPORTED!)](https://github.com/hikettei/cl-waffe) with a similar name. Don't misunderstand that: cl-waffe**2** is the latest one and all features are inherited from the old one.

cl-waffe2 provides fast, systematic, easy to optimize, customizable, and environment- and device- independent abstract matrix operations, and reverse mode tape-based Automatic Differentiation on Common Lisp. Plus, we also provide features for building and training neural network models, accelerated by JIT Compiler.

Roughly speaking, this is a framework for the graph and tensor abstraction without overheads. All features provided here can be extended by users without exception. And with the minimal code. cl-waffe2 is designed as the truly easiest framework to write extensions by users. There's no barrier between users and developers. There's no restriction imposed by framework ignoring Common Lisp.

Its abstraction layers are almost reaching the goals and enough practical, but there is still a serious lack of backend functionality, and documentations. Contributions are welcome and feel free to contact me: [hikettei](https://github.com/hikettei) if you've interested in this project.

## What's the next?

- [Setting up cl-waffe2](./install)

- [Step-by-step examples](./examples)

- [Learn more about its abstraction without overhead](./overview)

- [Sample Projects](https://github.com/hikettei/cl-waffe2/tree/master/examples)



# cl-waffe2, The JIT Compiler First Deep Learning Framework.

Since few month ago, I was working on the preceding project (deprecacted): [cl-waffe](https://github.com/hikettei/cl-waffe).

# The Project Structures

1. CLOS, MOP, Conditional Systemを最大限取り入れる

2. Defined-and-runだけど開発がしやすい

3. モデルの実行時は、エラーチェックなどを極力省く (C++/GGMLなどにCompile)

model.compile()前提のPyTorchのようなもの。

前回のプロジェクトから引き継ぎ:

`defnode` ついでに形状を記述するためのミニ言語などを取り入れる. backendなどのUIをもう少しまともにする

`defoptimizer` ... backend切り替えやすく
廃止する：

`defmodel` ... CLOSでおk

The Project is consisted of:
    `./source/vm/network/package.lisp` ... defnodeのmacro, UI, 形状検査, 形状検査のためのミニ言語 など...
    `./source/vm/compiler/defnodeから各Backendのためのコードを生成`
    `./source/apis/nn`
    `./source/apis/optimizer`
    `./source/apis/utils` ... torch.nnと同じような感じ, source/vmの上で成立
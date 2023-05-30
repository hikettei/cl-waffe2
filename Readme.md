
# cl-waffe2, Deep Learning Framework With Powerful Language, Common Lisp.

**The project is still in the concept stage.**

Since few month ago, I was working on the preceding project: [cl-waffe](https://github.com/hikettei/cl-waffe). I'm depcrecated with this project because:

1. It's just an imitation of PyTorch and I don't see the point of doing it on Common Lisp.

2. Define-by-run Style is cetrainly powerful for researching, but with regard to product, this feels unreasonable in my optinion. This is because the shape of the matrix at runtime is undetermined and therefore difficult to optimise.

3. Useful features such as CLOS and Conditional System should have been incorporated to the maximum extent possible, and the annoying (e.g.: Numpy's ShapingError) errors should have been made as clear as possible.

4. Difficult (quite laborious) to port to OpenBLAS CUDA Metal etc...

5. Make macro-based JIT a first priority mechanism. (tbh I'm inspired by CFFI's translating system)

# Features

### Safe and Static Shaping

No gc time, Static Memory Space...

### View First.

### Macro-Base JIT

### All Tensors and Nodes are Generic.

### Useful APIs

# The Project Structures


前回のプロジェクトから引き継ぎ:

`defnode` ついでに形状を記述するためのミニ言語などを取り入れる. backendなどのUIをもう少しまともにする

`defoptimizer` ... backend切り替えやすく
廃止する：

`defmodel` ... CLOSでおk

欲しい: waffe-viz (3d plotting etc)

```
The Project is consisted of:
    ./source/vm/generic-tensor/ ... 各バックエンドごとのTensorの定義 viewのパーサー 各バックエンドの言語にCompileする
    ./source/vm/nodes/package.lisp ... defnodeのmacro, UI, 形状検査, 形状検査のためのミニ言語 など...
    
    ./source/apis/nn
    ./source/apis/optimizer
    ./source/apis/utils ... torch.nnと同じような感じ, source/vmの上で成立
```

# Naming Rules

ソースコード内では全て`waffe`
パッケージを呼び出すときは`waffe2`

# Concepts

Shapeの制約をすっごく厳しくする

(compile node)で途中までノードを評価 (print hoge)で演算途中の結果を見れるようにする (e.g.: cl-waffeの(value tensor)と同じことをする)

```lisp
(defnode (AddNode (myself)
	  :where `([~] [~] -> [~])
	  :documentation "The Node Addnode Provides ..."))


(define-impl (AddNode :device CPUTensor)
	     :forward ((node x y)
		       (declare (ignore node))
		       (+ x y))
	     :backward ((node dy)
			))
```

# TODO

Documentの整理

set-variable

define-impl, 引数の数の違いをdetect

print-object(Node)

Broadcast

SVD

lparallel?

Github Actions

distributions

# Acknowledgements

Todo

1. Viewのテスト (strideの加算) at generic-tensor

2. View slice etc...

3. base-impl

4. construt-backward

5. Scalar and Matrix (Shape Generics)

Forward時Backward時の余計な計算ノードの枝刈り、memory-alloc strategy.

6. Parallel Programming?

TODO:

save-for-backward

scalar-XXX operations

defnode IN defnode

ViewNode (with facet-monopoly-mode)

NodeViz


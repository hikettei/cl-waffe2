
# Examples

## Template project

Due to its modularity, cl-waffe2 provides separated namespaces for separated features. If you don't care about that, you can follow the package below and all features become available.

```lisp
(in-package :cl-user)

(load "cl-waffe2.asd")    ;;
(ql:quickload :cl-waffe2) ;; These two steps should be included in asdf configuration.

(defpackage :your-project-template
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/nn
   :cl-waffe2/distributions
   :cl-waffe2/base-impl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/vm
   :cl-waffe2/optimizers

   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.lisp
   :cl-waffe2/backends.jit.cpu))

(in-package :your-project-template)
```

## Modeling

We use `AbstractNode` to represent the smallest unit of computation, `Composite` as a set of AbstractNode, `asnode` to treat functions as a Composite, and `defsequence` to compose several callable objects works like a arrow function. (In cl-waffe2, named as `call->`).

In this example, we demonstrate that Common Lisp Coding Style can be used to define Deep Learning Models in this elegant way.

### MLP

```lisp
(defsequence MLP-Sequence (in-features hidden-dim out-features
			   &key (activation #'!relu))
	     "Three Layers MLP Model"
	     (LinearLayer in-features hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim out-features))
```

### CNN

```lisp
(defsequence CNN ()
	     (Conv2D 1 4 `(3 3))
	     (asnode #'!relu)     
	     (MaxPool2D   `(2 2))
	     (Conv2D 4 16 `(5 5))
	     (asnode #'!relu)
	     (MaxPool2D `(2 2))
	     (asnode #'!reshape t (* 16 4 4)) 
	     (LinearLayer (* 16 4 4) 10))
```

### ResNet

```lisp
(Coming soon)
```

## Criterion

Several smaller nodes can be composed to create a larger function. Likewise Arrow function in Clojure, cl-waffe2 uses an util to compose multiple nodes. Once the following utilities have been defined, the code can be concisely written from then on. Note that this coding style is nothing other than what I like and readers do not necessarily have to follow it.

```lisp
(defun criterion (criterion X Y &key (reductions nil))
  (apply #'call->
	 (funcall criterion X Y)
	 (map 'list #'asnode reductions)))
```

```lisp
;; Usage

;; MSELoss
(criterion #'MSE pred excepted :reductions '(#'!sum #'->scal))

;; SoftmaxCrossEntropyLoss
(criterion #'softmax-cross-entropy pred excepted :reductions '(#'!sum #'->scal))
```

## Compiling Models

Since the laziness enables cl-waffe2 to optimize the network given futher information, you have to explict when you need to observe the results. The `build` function compiles given lazily evaluated neural network, returning `Compiled-Composite` which possess: compiled models(forward, backward), memory-pool, shape information, variable information, parameters.

```lisp
(defun compile-model (model X Y &key (lr 1e-3))
  (let* ((lazy-loss (criterion #'softmax-cross-entropy X Y			       
			       :reductions (list #'!sum #'->scal)))
	 (compiled-model (build lazy-loss :inputs `(:X :Y))))
    (mapc (hooker x (Adam x :lr lr)) (model-parameters compiled-model))
    (values compiled-model model)))
```

A Lazy Tensor needs to be created with `make-input` to inform the shape. This value can be changed later by giving a name to it and explict a list of names when building.

```lisp
;; Model
(compile-model (MLP-Sequence (* 28 28) 512 10)
               (make-input `(batch-size ,(* 28 28)) :X)
	       (make-input `(batch-size 10)         :Y))

;; CNN

;; ResNet

```

## Optimizing Models

We do not impose any constraints on how a training phase is implemented by users; You can write in a style that leans towards your favourite library, or you can create your original template! Enjoy the freedom of coding!

```lisp
(defun step-model (model X Y)
  (let ((act-loss (forward model X Y)))
    (backward model)
    (mapc #'call-optimizer! (model-parameters model))
    (tensor-vec act-loss)))

;; Example:
(step-model compiled-model (randn `(10 ,(* 28 28))) (randn `(10 10)))
```

## Training Models from actual data.

(TODO)

## Plotting

(TODO)

## Save trained weights

(TODO)
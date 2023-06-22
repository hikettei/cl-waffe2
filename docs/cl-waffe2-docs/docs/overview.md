
# Concepts

## Project Structure

cl-waffe2 consists of the following systems.

### Fundamental System

```
:cl-waffe2/vm.nodes
:cl-waffe2/vm.generic-tensor
```

`:cl-waffe2/vm.nodes` provides a system on constructing networks (e.g.: `AbstractNode` and `defnode`, `Composite` and `defmodel` etc...), and management to shape (e.g.: `:where pharse`, `subscript` etc...).

`:cl-waffe2/vm.generic-tensor` provides on the other hand a system on differentiable `AbstractTensor` (e.g.: `build` `make-tensor` `make-input` etc...)

All other packages are built on this package.

### Standard APIs

```
:cl-waffe2/base-impl
```

`:cl-waffe2/base-impl` provides a standard implementation for operations (e.g.: `!add` `proceed` `!view` etc...), including `defnode` definition and defun parts.

### Standard Backends/Implemenetations

```
:cl-waffe2/backends.lisp
:cl-waffe2/backends.cpu    
```

Both of them are standard implementation of `:cl-waffe2/base-impl` for CPU.

As of this writing (2023/06/20), cl-waffe2 has a only impls for CPU, however, If only time and money would permit, I'm willing to implement CUDA/Metal Backends.

:cl-waffe2/backends.lisp is `work enough` first, it is Portable (based on ANSI Common Lisp) and supports AVX2 but far from `full speed`.

On the other hand :cl-waffe2/backends.cpu is accelerated by OpenBLAS (maybe MKL is ok) and other foreign backends, this is SBCL-Dependant and sometimes could be unsafe, but provides `full speed`.


```
:cl-waffe2/backends.fastmath (NOT IMPLEMENTED YET!)
```

(TODO)

### Other Utils

```
:cl-waffe2 ;; Provides multi-threading APIs and config macros!
:cl-waffe2/nn ;; Provides Basic neural-network Implementations.
:cl-waffe2/optimizers ;; Provides Basic Optimizers
:cl-waffe2/viz ;; Provides Vizualizing APIs
etc...
```

### To Get Started!

It is recommended to load the system to be used when defining the package.

Please pick up the packages that you need depending on your needs.

For Example:
```lisp

(in-package :cl-user)

(defpackage :your-project-name
    (:use :cl
          :cl-waffe2
	  :cl-waffe2/vm.generic-tensor
	  :cl-waffe2/vm.nodes
	  :cl-waffe2/base-impl
	  :cl-waffe2/distributions
	  :cl-waffe2/backends.lisp
	  :cl-waffe2/backends.cpu
	  :cl-waffe2/nn
	  :cl-waffe2/optimizers
	  :cl-waffe2/viz))

(in-package :your-project-name)

;; Your code follows...

```

cl-waffe2 does not cause name clashes with other libraries or existing functions.
In addition, with regard to the function that generates the node is `!`, there is a rule that functions that create a node must start with `!` (ignoring the exception of the `proceed` function).

## Differentiable operations based on multiple backends

All operations in cl-waffe2 can be performed in the following form.

```
                           [AbstractNode]
	                             |
            |--------------------|------------------------|
 [CPU Implementation1] [CPU Implementation2] [CUDA Implementation1] ...
```

AbstractNode is a class declared via `defnode` macro, and each implementation is implemented via `define-impl` macro.

There can be more than one implementation for a single device. (e.g.: it is possible to have a normal implementation and an approximate implementation for the exp function).

One of the concepts is to minimise code re-writing by defining abstract nodes and switching the backends that executes them depending on the device they run on and the speed required.

As an example, consider implementing the addition operation `!add`.

The addition operation `AddNode` is the operation of finding the sum of two given matrices A and B and storing the result in A.

```lisp
(defnode (AddNode (myself)
            :where (A[~] B[~] -> A[~])
	    :documentation "A <- A + B"
	    :backward ((self dout dx dy)
	               (declare (ignore dx dy))
		       (values dout dout))
    ;; Here follows constructor's body
    ;; AddNode class initialized is passed as *myself*
    )
```

Here,

`:where` describes how matrices are performed. Before -> clause refers to the arguments, after -> clause refers to the shape of matrix after the operation.

It describes:

1. Takes A and B as arguments and returns a matrix of pointers of A
2. All matrices have the same shape before and after the operation.

`:backward` defines the operation of backward. This declaration can be made either in `defnode` or in `define-impl`, whichever you declare.

The declared node can be initialized using the function `(AddNode)`, but seems returning errors.

```lisp
(AddNode)
;; -> Couldn't find any implementation of AddNode for (CPUTENSOR LISPTENSOR).
```

This is because there is not yet a single implementation for `AddNode`, so let's define how AddNode works via `define-impl` macro.

One operation can be defined for a backend that can be declared by inheriting from the cl-waffe2/vm.generic-tensor:AbstractTensor class.

As of this writing(2023/06/20), cl-waffe2 provides following backends as standard.

1. LispTensor - Portable/Safety First, speed comes second. Everything works on ANSI Common Lisp.


2. CPUTensor - This is SBCL-dependant, but supports OpenBLAS linear-algebra APIs.

Of course, if necessary, you can create a new backend.

```lisp
(defclass MyTensor (AbstractTensor) nil)
```

(See also: [tensor.lisp](https://github.com/hikettei/cl-waffe2/blob/master/source/backends/lisp/tensor.lisp))

After defining a new backend, it is **NOT** necessary to give a re-implementation for all standard implementations in cl-waffe2. Select the appropriate backends in order of array compatibility.

Anyway, this is how AddNode is defined for LispTensor in cl-waffe2.

```lisp
(define-impl (AddNode :device LispTensor)
	     :forward ((self x y)
		       (let ((adder (matrix-add (dtype x))))
			 `(,@(call-with-view
			      #'(lambda (x-view
					 y-view)
				  `(funcall ,adder
			 		    (tensor-vec ,x)
					    (tensor-vec ,y)
					    ,(offset-of x-view 0)
					    ,(offset-of y-view 0)
					    ,(size-of x-view 0)
					    ,(stride-of x-view 0)
					    ,(stride-of y-view 0)))
			      `(,x ,y))
			   ,x))))
```

In `:forward` write the expansion expression for the operation in the same way as when defining a macro with `defmacro`. (see below for details).

Why define-impl takes such a roundabout approach?

1. To generate a fast code depending on the matrix size and data type at runtime.

2. To pre-calculate all Indexes

3. It is possible to generate, for example, C code without necessarily performing the same operations.

(I believe that ideas on this macro needed to be given more thoughts, indeed, this is ugly...)

Let's Perform operations with the defined AddNode!

```lisp
(forward (AddNode) (randn `(10 10)) (randn `(10 10)))
{CPUTENSOR[float] :shape (10 10) :named ChainTMP9412 
  :vec-state [maybe-not-computed]
  ((-0.33475596  1.0127474    -0.060175765 ~ 1.4573603    -0.987001    -1.0165008)                    
   (-0.045512    -0.17995936  0.23593931   ~ 0.8409552    2.6434622    -0.5789532)   
                 ...
   (0.13282542   1.9386152    0.16213055   ~ 0.4363958    0.8294802    -0.1558509)
   (1.1732875    -1.5769591   -1.2152125   ~ -0.2833903   -0.81108683  0.9846606))
  :facet :input
  :requires-grad NIL
  :backward <Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>}
```

Look at :vec-state, at that moment, the operation is still not done yet. The tensor displayed is the equivalent to the first argument.

In cl-waffe2, all operations are lazy-evaluated, being JIT-compiled/Optimized/Parallelized via `build`, or `proceed` function.

You would think that this style programming would make your task more complex, but don't worry, we provide APIs that is as close as possible to defined-by-run, and REPL-Friendly.

```lisp
(proceed (!add (AddNode) (randn `(10 10)) (randn `(10 10))))

;; proceed-time function measures execution time without compiling time.
(proceed-time (!add (AddNode) (randn `(10 10)) (randn `(10 10))))
Evaluation took:
  0.000 seconds of real time
  0.000014 seconds of total run time (0.000014 user, 0.000000 system)
  100.00% CPU
  30,512 processor cycles
  0 bytes consed
  
{CPUTENSOR[float] :shape (10 10) :named ChainTMP9447 
  :vec-state [computed]
  ((-1.5820543   2.2804832    -0.5613338   ~ 1.1143546    -1.3096298   -1.3756635)                    
   (-1.5208249   0.21621853   2.660368     ~ -1.032644    0.25917292   -1.9737494)   
                 ...
   (2.2557664    2.4791012    -0.04298857  ~ -1.2520232   1.8216541    -2.818116)
   (0.8615336    0.92017823   -0.25378937  ~ 0.9697968    -0.6300591   1.5660275))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

You can switch backends via `(with-devices (&rest devices) &body body)` macro seamlessly.

```lisp
(with-devices (LispTensor CPUTensor)
   ;; The priority of dispatching backends is: LispTensor(First) -> CPUTensor(Second)
   ;; If there's no any impls on LispTensor, use CPUTensor instead.
    (!add (randn `(10 10)) (randn `(10 10))))
```

## JIT compile, In-place optimizing

As I said `Everything is lazy-evaluated, and compiled`, JIT Compiling is a one of main idea of this project.

Mainly, this produces two benefits.

### Infinite number of Epochs, No Overheads of funcall.

As all lisper know, there is a unignorable overhead when calling methods.

```lisp
(defmethod test-method ((a fixnum) (b fixnum))
	(+ a b))

(defmethod test-method ((a single-float) (b single-float))
	(+ a b))

(time (dotimes (i 100000000) (test-method 1.0 1.0)))
Evaluation took:
  0.560 seconds of real time
  0.554936 seconds of total run time (0.551612 user, 0.003324 system)
  99.11% CPU
  1,291,693,656 processor cycles
  0 bytes consed

(defun test-fun (a b)
	(declare (type single-float a b))
	(+ a b))

;; Also, defun can be inlined at the end.
(time (dotimes (i 100000000) (test-fun 1.0 1.0)))
Evaluation took:
  0.298 seconds of real time
  0.297827 seconds of total run time (0.296968 user, 0.000859 system)
  100.00% CPU
  688,686,522 processor cycles
  0 bytes consed
```

In this project, which uses a large number of generic functions!, this overhead becomes non-negligible at every Epoch, especially when the matrix size is small.

Therefore, we took the approach of defining a new function by cutting out the necessary operations from the lazy-evaluated nodes, part by part.

;; cl-waffe2's benchmark

TODO: Update This section

```lisp
(let ((f (build (!sin 1.0))))
	(time (dotimes (i 100000) (funcall f))))

;; Fix: tensor-reset!'s overhead...
(defun test-f (x)
    (sin (sin (sin (sin x)))))

(time (dotimes (i 100000) (test-f 1.0)))
```

	
### In-place optimizing

This is a usual function in cl-waffe2, which finds the sum of A and B.

```lisp
(!add a b)
```

But internally, the operation makes a copy not to produce side effects.

```lisp
(forward (AddNode) (!copy a) b)
```

Without making a copy, the value of A would be destructed instead of having to allocate extra memory.

```lisp
(let ((a (make-tensor `(3 3) :initial-element 1.0)))
      (print a)
      ;; {CPUTENSOR[float] :shape (3 3)  
      ;;  ((1.0 1.0 1.0)
      ;;   (1.0 1.0 1.0)
      ;;   (1.0 1.0 1.0))
      ;;  :facet :exist
      ;;  :requires-grad NIL
      ;;  :backward NIL}
      ;; (eval A <- A + B)
      (proceed (forward (AddNode :float) a (randn `(3 3))))
      (print a)
      ;; {CPUTENSOR[float] :shape (3 3)  
      ;;  ((2.0100088   0.2906983   1.5334041)
      ;;   (-0.50357413 2.389317    0.7051847)
      ;;   (1.3005692   1.5925546   0.95498145))
      ;;   :facet :exist
      ;;   :requires-grad NIL
      ;;   :backward NIL} )

```

Operations that do not allocate extra space are called **in-place** (or sometimes destructive operations?).

Making operations in-place is a rational way to optimize your programs, but this is a trade-off with readability, because the coding style is more like a programming notation than a mathematical notation.

Let's take another example.

```math
out = f(Input) + f(f(Tensor))
```

(TODO)

```lisp
(defnode (1DFunc (self)
	  :where (A[~] -> A[~])))

(define-impl (1DFunc :device LispTensor)
	     :forward ((self x)
	               `(progn ,x))
	     :backward ((self dout dx) (values dout)))

(defun f (tensor)
    (forward (1DFunc) (!copy tensor)))
```

```lisp
(let ((k (!add (make-input `(3 3) nil) (f (f (randn `(3 3) :requires-grad t))))))

	(build k)
        (cl-waffe2/viz:viz-computation-node k "assets/1d_fn_arg.dot"))
```

### Before Optimized Vs After Optimized.

<img alt="bf" src="../../../assets/1d_fn_arg.png" width="45%">
<img alt="bf" src="../../../assets/1d_fn_arg_optimized.png" width="45%">

(TODO)


## Broadcasting APIs, View APIs

!flexible

!view

## REPL-Friendly function, proceed

Proceed

Proceed-backward

Proceed-time

## Strong Shaping APIs

syntax of :where pharse
Shape Error Reports

## Constructing networks with defnode/defmodel

defnode

defmodel

call

forward

## Data Structures

Parameter/Tensor/Input/ScalarTensor

## Optimizing Parameters

defoptimizer


Tutorials Over!

I'll keep my finger crossed.


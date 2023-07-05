
# A road to cl-waffe2

## Project Structure

Thank you for having interest in my project. Before we start the tutorial, let me explain the structure of cl-waffe2 package.

Mainly, cl-waffe2 consists of the following packages.

### Fundamental System

These two package form the basis of cl-waffe2:

```
:cl-waffe2/vm.nodes
:cl-waffe2/vm.generic-tensor
```

The package `:cl-waffe2/vm.nodes` provides a system for constructing neural networks, including `AbstractNode`, `Composite`, `Shaping API` etc...

On the other hand, `:cl-waffe2/vm.generic-tensor` provides features on `AbstractTensor`, including `JIT Compiler`, `NodeTensor`, `Memory-Pool` etc...

### Standard APIs

```lisp
(Figure: Dependencies of cl-waffe2)

                            [CPUBackend ]
            [base-impl] --- [LispBackend]
                |           [CUDABackend] ...
	            |
   [vm.generic-tensor] [vm.nodes]
   
```

```
:cl-waffe2/base-impl
```

Using the basic system of cl-waffe2, `:cl-waffe2/vm.generic-tensor` and `:cl-waffe2/vm.nodes`, the package `:cl-waffe2/base-impl` provides a standard implementation of matrix (sometimes scalar) tensor operations. The operation we say is including: `defun` parts, and abstract definition of operation.

Before we go any futher: cl-waffe2 is working on `AbstractTensor` (inspired in Julia's great idea, `AbstractArray`), which separates **implementation** of the operation from the
**definition.** In that respect, `:cl-waffe2/base-impl` provides the **definition** of operations, while the packages we about to mention provides **implementation** of operations.

### Standard Backends/Implemenetations

As of this writing (2023/07/05), we provide two standard implementation of `:cl-waffe2/base-impl`, both of them are working on CPU.

```
:cl-waffe2/backends.lisp
:cl-waffe2/backends.cpu    
```

If only time and money would permit, I'm willing to implement CUDA/Metal Backends.

:cl-waffe2/backends.lisp is `work enough` first, it is Portable (based on ANSI Common Lisp) and supports AVX2 but far from `full speed`.

On the other hand :cl-waffe2/backends.cpu is accelerated by OpenBLAS (maybe MKL is ok) and other foreign backends, this is SBCL-Dependant and sometimes could be unsafe, but provides `full speed`.


TODO:

```
:cl-waffe2/backends.fastmath (NOT IMPLEMENTED YET!)
```

(TO BE) Supporting vectorized mathematical functions, AVX512 instructions.

### Neural Network

```lisp
:cl-waffe2/nn ;; Provides Basic neural-network Implementations.
:cl-waffe2/optimizers ;; Provides Basic Optimizers
```

### Utils

```lisp
:cl-waffe2     ;; Provides multi-threading APIs and config macros!
:cl-waffe2/viz ;; Provides Vizualizing APIs of computation node
etc...
```

### To Get Started!

If you're going to start with defining a new package, It is recommended to `:use` the package to be used.

Read the description above and select and describe the packages you think you need. (or you can just copy and paste it.)

This is an example case of `:your-project-name` package.
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

If you're working with REPL (or new to Common Lisp?), you can try cl-waffe2 features like this:


```sh
$ ros run
> (load "cl-waffe2.asd") # cl-waffe2.asd should be placed where SBCL can read it.
> (ql:quickload :cl-waffe2)
> (in-package :cl-waffe2-repl) ;; this is a playground place, and all features are available
```

The tutorials below should be also working on REPL, (indeed, cl-waffe2 is REPL-friendly!), you can learn how cl-waffe2 works by copying and pasting the example codes.

## Basic: Building Computation Nodes Lazily

Since `Do not run until the node is optimized` is a one of cl-waffe2 policy, all operations in cl-waffe2 is lazy evaluation unless defined by a special macro.

Therefore, calling `!add` function which finds a sum of given arguments, the retuend tensor isn't still computed, but `:vec-state` = `[maybe-not-computed]`

```lisp
(!add 3.0 2.0)
```

```lisp
{SCALARTENSOR[float]  :named ChainTMP23305 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (1) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: SCALARANDSCALARADD-SCALARTENSOR (A[SCAL] B[SCAL] -> A[SCAL]
                                                    WHERE SCAL = 1)>}
```

`:vec-state` indicates the computation state of tensor, and it says exactly what it says.

You can continue the operation by connecting the returned tensor and next operation.

For example, the figure below in cl-waffe is representece as:

```math
out = 3 + 2 * 4
```

```lisp
(defparameter out (!add 3 (!mul 2 4))) ;; out <- 3 + 2 * 4
```

To obtain the state in which the operation is performed, calling the function `(proceed toplevel)` is a vaild option.

```lisp
(proceed out)

{SCALARTENSOR[int32]  :named ChainTMP28326 
  :vec-state [computed]
    11
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

`proceed` is a differentiable operation which instantly compiles and executes all the previous node of `toplevel`. In addition, there's another way to accept nodes: `(build out)` or `(define-composite-function)`, but they're a little complicated, so explained in the other sections.

The moment compiling function is called, cl-waffe2 prunes all unused copying, computes all `View Offsets`, schedules memory allocation and (Currently it's not working though) multi-threading.

## AbstractTensor - One operation, Multiple implementations.

### Background

```lisp
(Operations in cl-waffe2)

                           [AbstractNode]
	                             |
            |--------------------|------------------------|
 [CPU Implementation1] [CPU Implementation2] [CUDA Implementation1] ...
```

`Julia` has introduced [AbstractArray](https://docs.julialang.org/en/v1/base/arrays/) in their libraries, separating the common (generic) parts of the array from each backend implementation. Since `AbstractTensor` increased portability between devices on which they run (even on CPU!), cl-waffe2 wholly introduced this feature.


In cl-waffe2, The generic definition of operations, `AbstractNode` is a class declared via the `defnode` macro, and depending on the devices we use, the `define-impl` macro defines an implementation.

Conveniently, there can be more than one implementation for a single device. (e.g.: it is possible to have a normal implementation and an approximate implementation for the exp function on single CPU).

One of the policy is to minimise code re-writing by defining abstract nodes and switching the backends that executes them depending on the device they run on and the speed required.


### Example: AddNode

Here's an example of how I've implemented the operation `!add`.

`AddNode-Revisit` is `AbstractNode` of finding the sum of two given matrices A and B and storing the result in A. Here's the segment from the source code.

```lisp
;; Reimplementation of AddNode
(defnode (AddNode-Revisit (myself dtype)
            :where (A[~] B[~] -> A[~])
	        :documentation "A <- A + B"
	        :backward ((self dout dx dy)
	                   (declare (ignore dx dy))
		               (values dout dout))))
```

`AbstractNode` is a CLOS class with the following operation.

1. shape changes before and after the operation, and which pointer to use? is described in `:where`. Before `->` clause refers to the arguments, after `->` clause refers to the shape of matrix after the operation.

It says:

1. Takes A and B as arguments and returns a matrix of pointers of A
2. All matrices have the same shape before and after the operation.

Also, `:backward` defines the operation of backward. This declaration can be made either in `defnode` or in `define-impl`, whichever you declare.

The declared node can be initialized using the function `(AddNode-Revisit dtype)`, but seems returning errors.

```lisp
(AddNode-Revisit :float)
;; -> Couldn't find any implementation of AddNode for (CPUTENSOR LISPTENSOR).
```

This is because there is not yet a single implementation for `AddNode-Revisit`.

One operation can be defined for a backend that can be declared by extending the `cl-waffe2/vm.generic-tensor:AbstractTensor` class. Here's `LispTensor`, and `CPUTensor`, and of course, if necessary, you can create a new backend like:

```lisp
(defclass MyTensor (AbstractTensor) nil)

;; Initializer/Allocator
(defmethod initialize-instance :before ((tensor MyTensor)
					&rest initargs
					&key &allow-other-keys)
  ;; if projected-p -> alloc new vec
  (let* ((shape (getf initargs :shape))
	 (dtype (dtype->lisp-type (getf initargs :dtype)))
	 (vec   (getf initargs :vec))
	 (facet (getf initargs :facet))
	 (initial-element (coerce (or (getf initargs :initial-element) 0) dtype)))
    (when (eql facet :exist)
      (if vec
	  (setf (tensor-vec tensor) vec)
	  (setf (tensor-vec tensor)
		(make-array
		 (apply #'* shape)
		 :element-type dtype
		 :initial-element initial-element))))))

;; If data storage is differ from CL Array, override vref and (setf vref) method.
```

(See also: [tensor.lisp](https://github.com/hikettei/cl-waffe2/blob/master/source/backends/lisp/tensor.lisp))

The devices to use can be switched `with-devices` macro.

```lisp
(with-devices (MyTensor LispTensor) ;; The further to the left, the higher the priority.
    (make-tensor `(10 10)))

;; -> MyTensor is created
{MYTENSOR[float] :shape (10 10)  
  ((0.0 0.0 0.0 ~ 0.0 0.0 0.0)           
   (0.0 0.0 0.0 ~ 0.0 0.0 0.0)   
        ...
   (0.0 0.0 0.0 ~ 0.0 0.0 0.0)
   (0.0 0.0 0.0 ~ 0.0 0.0 0.0))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

`MyTensor` has no implementation of any operations, but the code below is working.

```lisp
(with-devices (MyTensor LispTensor)
    (proceed (!add (randn `(3 3)) (randn `(3 3)))))

{MYTENSOR[float] :shape (3 3) :named ChainTMP28398 
  :vec-state [computed]
  ((-1.4494231  1.0320233   -1.8852448)
   (1.0886636   -0.37185743 0.99227524)
   (2.2778857   -0.82929707 2.3525782))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

This is because `MyTensor` and `LispTensor` are pointer compatible, and `AddNode` for `LispTensor` is used instead of undefined implementation, `AddNode` for `MyTensor`.

Of cource



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


### Lazy Evaluation

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
f(x) = sin(MaybeCopy(x))
```

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


## Optional Broadcasting, and View APIs

!flexible

!view

## Proceed, Build, Composite-Function

Proceed

Proceed-backward

Proceed-time

Composite-Function

## Shaping API with DSL

syntax of :where pharse
Shape Error Reports

## Basic Unit: AbstractNode and Composite

defnode

defmodel

call

forward

Composite

## Multiple facet of Tensor

Parameter/Tensor/Input/ScalarTensor

## Optimizing Model Parameter

defoptimizer


Tutorials Over!

I'll keep my finger crossed.


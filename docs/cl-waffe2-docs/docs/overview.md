
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

(P.S: `:cl-waffe2/backends.jit.lisp` is now partially available (not tested all), it is still unstable but demonstrates how to extend the JIT compiler on other backends in cl-waffe2.)

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

The lisp code below demonstrates an example case of `:your-project-name` package.

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

Therefore, calling `!add` function which finds a sum of given arguments, the retuend tensor isn't still computed, only setting `:vec-state` = `[maybe-not-computed]`.

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

For example, the figure below in cl-waffe is represented as:

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


In cl-waffe2, The generic definition of operations, `AbstractNode` is a class declared via the `defnode` macro, and depending on the devices we're working on, the `define-impl` macro defines an implementation.

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

One operation can be defined for a backend that can be declared by extending the `cl-waffe2/vm.generic-tensor:AbstractTensor` class. Here's `LispTensor`, and `CPUTensor`, and of course, if necessary, you can create a new backend `MyTensor` by just copying them:

```lisp
;; 1. Creating from AbstratTensor.
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


;; 2. Using an existing backend.


;; If you want to use a backend that is already implemented, the following line of code is sufficient.

(defclass MyTensor (CPUTensor) nil) ;; Adding a new backend is all done in this code!
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

Therefore, after defining a new backend, it is **NOT** necessary to give a re-implementation for all standard implementations in cl-waffe2. Select the appropriate backends in order of array compatibility.

The macro `define-impl` adds a new implementation of `device`.

```lisp
;; The code below is NOT working on REPL, but working in :cl-waffe2/backends.lisp package

(define-impl (AddNode-Revisit :device MyTensor)
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

In `:forward`, write the expansion expression for the operation in the same way as when defining a macro with `defmacro`. The `call-with-view` function is a general-purpose function to iterate the given tensor with computing offsets.

(P.S.: I believe that ideas on this macro needed to be given more thoughts, indeed, this is ugly... but I guess `composite-function` can be install without writing macros, not tested.)

The forward definition of node can be called with `(forward node &rest inputs)` function.

```lisp
(forward (AddNode :float) (randn `(10 10)) (randn `(10 10)))
{CPUTENSOR[float] :shape (10 10) :named ChainTMP28407 
  :vec-state [maybe-not-computed]
  ((-0.93102205  -0.25396287  0.45237574   ~ 0.54063225   0.56266963   -0.77444124)                    
   (-0.55870235  -0.9794068   -0.21233901  ~ 1.1901267    -0.83241004  -0.69876736)   
                 ...
   (-0.5366255   -0.9118863   1.274197     ~ 0.19851275   0.21501832   1.064277)
   (-0.65124494  0.15393624   -0.6625119   ~ -1.1875637   -2.007647    0.5431197))
  :facet :input
  :requires-grad NIL
  :backward <Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>}
```

Closely Looking at :vec-state, it says the operation isn't done yet. The embodied elements are displayed but this is because `AddNode` is defined as in-place operation, returning the first argument.

To accept this state instantly, we can use `proceed`.

```lisp
(proceed (forward (AddNode :float) (randn `(10 10)) (randn `(10 10))) :measure-time t)
Proceed-Time: First Trying
Evaluation took:
  0.000 seconds of real time
  0.000028 seconds of total run time (0.000019 user, 0.000009 system)
  100.00% CPU
  26,990 processor cycles
  0 bytes consed
  
Proceed-Time: Second Trying
Evaluation took:
  0.000 seconds of real time
  0.000003 seconds of total run time (0.000003 user, 0.000000 system)
  100.00% CPU
  6,300 processor cycles
  0 bytes consed
  
{CPUTENSOR[float] :shape (10 10) :named ChainTMP28477 
  :vec-state [computed]
  ((2.843876    2.3477855   3.3252454   ~ -1.0901415  -1.211004   -2.268893)                   
   (-2.7236757  -0.60536575 -0.61465085 ~ 2.383132    -0.22351071 -0.6449351)   
                ...
   (-0.7634125  0.7340392   2.7052975   ~ 1.1768849   3.609434    -1.3465445)
   (4.1204114   3.696868    -2.1895533  ~ -1.5550013  2.6361299   0.31319892))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```


## Compiling, and In-place optimizing

### Compiled Model

Compiling Common Lisp Code at runtime is certainly fast, but isn't enough. In order to re-use compiled nodes, there is `Compiled-Composite` class to manage the state.

`Compiled-Composite` can be obtained by calling `(build toplevel)`

```lisp
(let* ((out (!sum (!add (randn `(10 10)) (randn `(10 10)))))
       (compiled-model (build out)))
    compiled-model)

<Compiled-Composite
    forward:  #<FUNCTION (LAMBDA ()) {53D7ED1B}>
    backward: #<FUNCTION (LAMBDA ()) {53D4D78B}>

+= [Tensors in the computation node] =======+

Subscripts:


Variables:
 NAMES |  SIZE | 


 - The number of tmp variables : 15
 - The number of parameters    : 0
+========================================+
>
```

`(forward compiled-composite)`, `(backward compiled-composite)` calls forward/backward functions respectively.

```lisp
(let* ((out (!sum (!add (randn `(10 10)) (randn `(10 10)))))
       (compiled-model (build out)))
     (print (forward compiled-model))
     (print (backward compiled-model)))

{CPUTENSOR[float] :shape (1 1) -> :view (<0> <0>) -> :visible-shape (1 1) :named ChainTMP29625 
  ((-24.876368))
  :facet :input
  :requires-grad NIL
  :backward NIL} 
T 
```

But what if one wants to change the value of first argument? Replace `(make-tensor)` to be replaced later with a `(make-input)` function.

```lisp
(make-input shape input-name)
```

`shape` can include symbols, to be determined later.

```lisp
(let* ((out (!sum (!add (make-input `(a b) :InputA) (randn `(10 10)))))
       (compiled-model (build out)))
     compiled-model)
     
<Compiled-Composite
    forward:  #<FUNCTION (LAMBDA ()) {53D9AF7B}>
    backward: #<FUNCTION (LAMBDA ()) {53D9D53B}>

+= [Tensors in the computation node] =======+

Subscripts:
     [A -> ?, max=?]
     [B -> ?, max=?]


Variables:
 NAMES  |  SIZE  | 
––––––––––––––––––
 INPUTA |  (A B) | 


 - The number of tmp variables : 15
 - The number of parameters    : 0
+========================================+
>
```

The function `(make-input)` itself, doesn't have a vector storage. (as long as `(tensor-vec tensor)` function isn't called). Accordingly, someone has to **embody** the storage of InputTensor with `ExistTensor`.

`(set-input compiled-composite input-name actual-tensor)` embodies given InputTensor in the computation node with actual-tensor.

```lisp
(let* ((out (!sum (!add (make-input `(a b) :InputA) (randn `(10 10)))))
       (compiled-model (build out)))

    (set-input compiled-model :InputA (randn `(10 10)))
    (print (forward compiled-model))
    (print (backward compiled-model))

    (set-input compiled-model :InputA (randn `(10 10)))
    ;; ... working on another input
    )

{CPUTENSOR[float] :shape (1 1) -> :view (<0> <0>) -> :visible-shape (1 1) :named ChainTMP29804 
  ((17.631124))
  :facet :input
  :requires-grad NIL
  :backward NIL} 
T 
```

### In-place optimizing

This is a usual function in cl-waffe2, which finds the sum of A and B.

```lisp
(!add a b)
```

However, this is how `!add` is defined internally. This makes a copy twice times not to make side effects.

```lisp
;; In source/base-impl/arithmetic.lisp

(forward (AddNode dtype) (!copy a) (!copy b))
```

Without copying, the content of `a` is overwritten:

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

To put it bluntly, it is natural to think this copy is just a waste of memory. However, In this case, disabling `!copy` is a rational way to optimize the performance of the program. (i.e.: replace with in-place operation).

Owing to lazy evaluation of cl-waffe2, unnecessary `(!copy)` operation can be deleted automatically by checking the number of tensor references in a node.

Let f(x) be a operation defined as:

```math
f(x) = sin(MaybeCopy(x))
```

Let the computation node be below:

```math
out = f(Input) + f(f(Tensor))
```

Formulating the same network in cl-waffe2:

```lisp
;; (Let me define the utilities to be used in defnode in advance)

;; Tips:
;; Obtain function of :lazy-evaluation -> immediate execution.

(defmodel (SinModel (self)
             :where  (X[~] -> [~])
             :on-call-> ((self x)
	                     (declare (ignore self))
			             (!sin x))))
	     
(define-composite-function (SinModel) !sin-static :dtype :float)

;; (!sin-static (randn `(10 10))) is instantly executed. not lazy-evaluated.
```

```lisp
;; Basic Units in the network:

;; General Definition of f(x)
(defnode (F-Node (self)
          :documentation "f(x) = sin(x)"
	      :where (A[~] -> A[~])))

;; Implementation of f(x)
;; Setting :device = t, -> the impl is working on all devices.

(define-impl (F-Node :device t)
	     :forward ((self x) `(!sin-static ,x))
	     :backward ((self dout dx) (values (!mul dx (!cos dout)))))

;; The caller of f(x)

(defun !f (x)
    (forward (F-Node) (!copy x)))
```

Through `:cl-waffe2/viz` package, we can visualize how the operation is performed.

```lisp
;; (make-input ... nil): creates a caching tensor, being the elements of it isn't guaranteed to be 0.0.
(let ((k (!add (make-input `(3 3) nil)
               (!f (!f (randn `(3 3) :requires-grad t))))))
        (cl-waffe2/viz:viz-computation-node k "assets/bad_node.dot")
	    (build k) ;; optimized
           (cl-waffe2/viz:viz-computation-node k "assets/opt_node.dot"))
```

The result is written in `dot language`.

```sh
$ dot -Tpng ./assets/bad_node.dot > ./assets/bad_node.png
$ dot -Tpng ./assets/opt_node.dot > ./assets/opt_node.png
```

### Before Optimized Vs After Optimized.

<img alt="bf" src="https://github.com/hikettei/cl-waffe2/blob/master/docs/cl-waffe2-docs/docs/assets/bad_node.png?raw=true" width="45%">
<img alt="bf" src="https://github.com/hikettei/cl-waffe2/blob/master/docs/cl-waffe2-docs/docs/assets/opt_node.png?raw=true" width="45%">

`ExistTensor` (created by `make-tensor`, or tensors whose requires-grad=t) is never overwritten.


## Network Units: Node and Composite

In this section, we learn the two key units, `Node` and `Composite`, to construct neural networks in cl-waffe2.

### Node and Composite

`Node(AbstractNode)` is the smallest unit of operation with forward and backward propagation. Its abstract definition is defined by a `defnode` macro, and It is implemented by a `(define-impl)` macro. The defined node is invoked by `(forward node &rest inputs)` function, at the same time, computation nodes are constructed.

```lisp
(defnode (SinNode-Revisit (self)
            :where (X[~] -> X[~])
	    :save-for-backward (t)
	    :backward ((self dout x)
	               (values (!mul dout (!cos x))))))

(define-impl (SinNode-Revisit :device t)
       :forward ((self x) `(!sin-static ,x)))


(forward (SinNode-Revisit) (randn `(10 10)))

{CPUTENSOR[float] :shape (10 10) :named ChainTMP32968 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (10 10) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: SINNODE-REVISIT-T (X[~] -> X[~])>}

(proceed *)

{CPUTENSOR[float] :shape (10 10) :named ChainTMP32955 
  :vec-state [computed]
  ((0.43090457   -0.24942507  -0.99978673  ~ 0.97256666   -0.9993819   0.37133723)                    
   (0.050297778  -0.048203766 0.11011651   ~ -0.28100008  -0.89788723  0.12841338)   
                 ...
   (0.32419643   0.15791988   -0.95573443  ~ 0.079026684  -0.4924342   0.99993217)
   (-0.04615228  -0.2262427   -0.6637178   ~ 0.8855889    -0.72787035  -0.65471023))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

On the other hand, `Composite` is a unit made up of several `Nodes`, defined by a `defmodel` macro. `(call model &rest inputs)` method invokes the `on-call->` form lazily, being compiled in the same way as nodes. Moreover, the defined `Composite` also can define a function for immeditate function by using the macro, `define-composite-function`. The behaviour is similar to `TorchScript`, cl-waffe2 traces the computation node, calling `(build toplevel)` and defines a `Composite-function`.

```lisp
(defmodel (Softmax-Model (self)
	   :where (X[~] -> [~]) ;; :where for Composite is optional!
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (let* ((x1 (!sub x (!mean x  :axis 1 :keepdims t)))
	                      (z  (!sum   (!exp x1) :axis 1 :keepdims t)))
                           (!div (!exp x1) z)))))

;; won't be evaluated until proceed/build is called.
(call (Softmax-Model) (randn `(10 10))

(proceed *)

			 
{CPUTENSOR[float] :shape (10 10) :named ChainTMP6184 
  :vec-state [computed]
  ((0.29810402  0.11953584  0.16032213  ~ 0.033787794 0.01729085  0.03808046)                   
   (0.032921903 0.085420445 0.10371924  ~ 0.06863596  0.10435363  0.07114864)   
                ...
   (0.23044951  0.14320189  0.16871664  ~ 0.019123536 0.03614414  0.10644407)
   (0.0377036   0.034945846 0.28327137  ~ 0.07359542  0.40399343  0.020138593))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}

;; Works at toplevel
(define-composite-function (Softmax-Model) !softmax-static)

;; No overheads of compiling, but there's a little overhead to dispatch the method.
(time (!softmax-static (randn `(10 10))))
Evaluation took:
  0.000 seconds of real time
  0.000301 seconds of total run time (0.000253 user, 0.000048 system)
  100.00% CPU
  691,808 processor cycles
  32,496 bytes consed
  
{CPUTENSOR[float] :shape (10 10) :named ChainTMP6195 
  ((0.042827643  0.13156936   0.06729175   ~ 0.059296332  0.17645036   0.04613843)                    
   (0.32095885   0.030778391  0.091331415  ~ 0.09311637   0.28322798   0.040707175)   
                 ...
   (0.045369238  0.045168925  0.12002338   ~ 0.2656273    0.01337298   0.41475114)
   (0.020064427  0.01839381   0.013036524  ~ 0.20158055   0.3377756    0.061546378))
  :facet :input
  :requires-grad NIL
  :backward NIL}
```

### Proceed vs Composite-function

Compared to `Proceed`, `Composite-function` and codes which consisted of it have a small overhead in calling a function, but it becomes negligible as the matrix size increases.

```lisp
;; Proceed
;; 1 * 1 Matrix
(let ((a (ax+b `(1 1) 0 1)))
    (proceed-time (!sin a)))
Proceed-Time: First Trying
Evaluation took:
  0.000 seconds of real time
  0.000077 seconds of total run time (0.000054 user, 0.000023 system)
  100.00% CPU
  141,818 processor cycles
  0 bytes consed
  
Proceed-Time: Second Trying
Evaluation took:
  0.000 seconds of real time
  0.000007 seconds of total run time (0.000006 user, 0.000001 system)
  100.00% CPU
  11,790 processor cycles
  0 bytes consed

;; Proceed
;; 1000 * 1000 Matrix

(let ((a (ax+b `(1000 1000) 0 1)))
    (proceed-time (!sin a)))
Proceed-Time: First Trying
Evaluation took:
  0.019 seconds of real time
  0.019118 seconds of total run time (0.017191 user, 0.001927 system)
  100.00% CPU
  44,118,196 processor cycles
  8,000,032 bytes consed
  
Proceed-Time: Second Trying
Evaluation took:
  0.015 seconds of real time
  0.015628 seconds of total run time (0.015613 user, 0.000015 system)
  106.67% CPU
  36,025,586 processor cycles
  0 bytes consed
```


```lisp
;; Composite-Function
;; 1 * 1 Matrix
(let ((a (ax+b `(1 1) 0 1)))
     (time (!sin-inline a)))
Evaluation took:
  0.000 seconds of real time
  0.000103 seconds of total run time (0.000098 user, 0.000005 system)
  100.00% CPU
  231,840 processor cycles
  0 bytes consed
  
;; Composite-Function
;; 1000 * 1000 Matrix
(let ((a (ax+b `(1000 1000) 0 1)))
     (time (!sin-inline a)))
Evaluation took:
  0.015 seconds of real time
  0.015862 seconds of total run time (0.015813 user, 0.000049 system)
  106.67% CPU
  36,632,326 processor cycles
  0 bytes consed
```

(Tips: the `call` method is designed to invoke `Composite`, but it is also applicatable into `AbstractNode`, that is, `call` is a general-purpose method to invoke nodes.)

### Sequence Model

Since the shape of matrices is declared everywhere operation, cl-waffe2 can trace the structure of neural networks lazily, and being checked before the execution.

In the code below, `defsequence` is a macro to define `Composite` sequentially, `(asnode function)` is a macro which coerce function into `Composite`.

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

 All composites/nodes that used to define `MLP-Sequence` has also a definition of shape.
	     
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

Not an operation is performed, nor a matrix is allocated at the moment `MLP-Sequence` is initialized, but done when compiling/invoking the computation node.

## Shaping API with DSL

See also: [Introducing Subscript DSL](../nodes/#introducing-subscript-dsl)

## View APIs

(TODO)

### Optional Broadcasting

In cl-waffe2, operations with several arguments must be called with the same shape of tensors as `:where` says. In the code below, since `!add` is declared as `A[~] B[~] -> A[~]`, the first and second argument, must have the same shape, same ranks. However, opeartions isn't always performed within the same ranks. In practice, `!add` isn't always used as just an element-wise operation because the total elements of tensor can be found via `!add`, adding biases to the given tensor is also realised by using `!add`. Indeed, `broadcasting` is a convenient operation when expressing matrix iterations without using `(loop for ...)`.

```lisp
(!add (randn `(3 3)) (randn `(3)))
; Evaluation aborted on #<CL-WAFFE2/VM.GENERIC-TENSOR:SHAPING-ERROR {100376BD13}>.
```

The same code would being broadcasted well and works on libraries which supports `Numpy Semantics Broadcasting`, but not working on cl-waffe2 as you can see.

This is because cl-waffe2 do not support `automatically broadcasting` but support `manually broadcasting`. That is, each place broadcast is needed, you also have to declare the tensor is broadcasted, since the condition of `broadcastable` is less restrictive which sometimes produce an unintended behaviour with no any errors even though broadcasting is only used in a limited situation.

`Numpy Semantic Broadcasting` has two rules:

1. Rank up: If matrices with different number of axes are called in a one operation, `one` is added to the tensor with smallest ranks to straighten up the number of dimensions.

2. Repeating 1: If the dimension at corresponding position do not match, and either one is `1`. `1` is repeated with the other.

There are two corresponding operations in cl-waffe2:

```
(Two main parts of broadcasting)
   (<1 x N> 1 1)
     ↑       ↑
  !flexible !view
```

```lisp
(!flexible (randn `(1 1)))

{CPUTENSOR[float] :shape (<1 x N> 1 1) :named ChainTMP2614 
  :vec-state [maybe-not-computed]
  ((0.6495824))
  :facet :input
  :requires-grad NIL
  :backward <Node: FLEXIBLE-RANK-NODE-T (A[~] -> A[~])>}
```

```lisp
(!view (randn `(1)) `(:broadcast 100))

{CPUTENSOR[float] :shape (1) -> :view (<(BROADCAST 100)>) -> :visible-shape (100) :named ChainTMP4406 
  :vec-state [maybe-not-computed]
  (-0.5008113 -0.5008113 -0.5008113 ~ -0.5008113 -0.5008113 -0.5008113)
  :facet :input
  :requires-grad NIL
  :backward <Node: VIEWTENSORNODE-T (A[RESULT] B[BEFORE] -> A[RESULT])>}
(0)

```

The function `(!flexible)` adds the `broadcastable dimensions` of the given tensor. In `<1 x N>` parts, 1 is repeated, 1 is added if any. In `1` parts, never broadcasted.


```lisp
(!view a `(:broadcast 10))
```

Mem:

If both of given tensors is broadcasted, we may need to make a copy to store the result since there's no array of broadcasted size.

This explicts: in which tensor, is broadcasting applied?, that is, there's more likely to useless copy is also removed. in-place broadcasting.

### Case1 - To higher, Batched Operation

(!matmul (!flexible ...) (!flexible ...))

### Case2 - To lower,  add biases to columns

(!add a (!view x ...))

TODO: `(with-broadcasting (a1 b1 (a b)) ...)` macro.

### Multidimensional Offsets.

(TODO)

## Optimizing Model Parameter

(TODO)

defoptimizer

deftrainer

parameter

Tutorials Over!

(TO ADD: ./Examples, training MNIST, Image processing, NLP etc...)

I'll keep my finger crossed.


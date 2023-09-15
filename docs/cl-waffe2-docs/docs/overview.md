# Programmable Deep Learning Framework

In the recent years, the widely accepted libraries and frameworks in the field of data science, such as matrix operations and mathematical optimization, have concentrated on popular and attractive programming languages: Python and Julia. But did you know that a long time ago (about 40~50 years), artificial intelligences are all about Lisp? At that time, research into artificial intelligences using symbolic logics was very popular, and Lisp and Prolog were the languages of choice. As the time went on, however, such research became less and less common, and instead languages that could easily handle large matrices became mainstream, which is roughly the history to date. However, even without the influence of symbolic theory, Common Lisp is still powerful language and indeed I'm one of Lispers who believes the advantages of applying this into data science. In this section, I will explain the concepts and benefits of my project cl-waffe2, an elegant and extensible Deep Learning Framework on Common Lisp.

## Why not: Keep Using Python?

- Python is not my cup of tea because of these reasons:
    - Verbose: Python Syntax is very simple and loved by a lot of engineers across their fields, but it's too simple; I'm always annoyed with a ton of decorators and redundant class initialization syntax.
    - Common Lisp is an envolving language under the standardization by ANSI Common Lisp established in 25 years ago; Writing an extension is no longer pain.
    - CL has more; Error Handlings, Dynamic Scopes, MetaProgramming, and more!
    - Ease of Debugging: For Lisp, REPL is their cup of tea!

I won't talk too much about this topic here since it has been discussed a lot elsewhere. In fact, however, the equivalent PyTorch code can be rewritten so short and intuitive using cl-waffe2.

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

(Reference: [BUILD THE NEURAL NETWORK](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html))

```lisp
(defsequence NeuralNetwork ()
	     "My Neural Network"
	     (asnode #'!flatten)
	     (LinearLayer (* 28 28) 512)
	     (asnode #'!relu)
	     (LinearLayer 512 512)
	     (asnode #'!relu)
	     (LinearLayer 512 10))
```

This is not only the case of defining networks but: When writing an extension, defining models, composing several functions, and visualizing... cl-waffe2 is designed to make more codes smaller and obvious!

Readers may well feel "How about Julia?" - Yes, I think Julia ecosystems are great, and cl-waffe2 is influenced by Julia everywhere. And, this should be introduced into Common Lisp, not just a few languages!

## Let's Get Started!

Note that all sample codes are working under this package:

```lisp
(defpackage :concepts
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
   :cl-waffe2/backends.lisp))
   
(in-pacakge :concepts)
```

## Nodes are Abstract, Lazy, and Small.

In neural networks, most temporary tensors aren't observed during training time while they actually consist a half of memory usage. Programmers have to take into account which tensors need intermediate values and should be in place - but If only compilers know the start and end points of the computation node, a simple algorithm which just counts up the reference count, can alternative this process. Likewise theano, and other existing libraries (e.g.: `Petalisp` ...) cl-waffe2 is nothing but Domain Specific Language, which represents the graph being compiled and optimized. I think I don't have the enough background to introduce Theano, but cl-waffe2 is also influenced by this framework; operations are lazy-evaluated, and later compiled. Accepting Lazy-Evaluation, that is, Nodes also have to imply the transformation of shapes, and which tensors should be in-place. These can be expressed in a single line using `Subscript DSL` macros (compiled in the toplevel).

In the simplest case, the AbstractNode `MatmulNode-Revisit` which is a reimplementation of existing `MatmulNode` and computes matrix multiplication of given two tensors: A and B, storing the result into C, can be declared like:

```lisp
(defnode (MatMulNode-Revisit (self dtype)
	  :where (A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :backward ((self dout a b c)
		     (declare (ignore c))
		     (values
		      (!matmul dout (!t b))
		      (!matmul (!t a) dout)
		      nil))
	  :documentation "OUT <- GEMM(1.0, A, B, 0.0, C)"))
```

In term of performance, ranks and shapes are declared anywhere. So you MUST understand this notation and specification if you want to work on cl-waffe2: (Specs: [representing-shapes-before-and-after-the-operation](https://hikettei.github.io/cl-waffe2/nodes/#representing-shapes-before-and-after-the-operation))

The macro `defnode` defines `AbstractNode` which guarantees the same behaviour across different devices. If you want to implement `MatmulNode-Revisit` in various devices; `CPU, Lisp, CUDA, Metal`, that should be defined with its device name, a rank of operations, elsewhere.

Here we have a function which calculates `GEMM` in Lisp Code, let's see how cl-waffe2 can use this.

```lisp
;; A_ii,B_ik->C_jk
;; The performance would be the worst. Should not be used for practical.
(defun gemm! (m n k a-offset a b-offset b c-offset c)
  "Computes 1.0 * A[M N] @ B[N K] + 0.0 * C[M K] -> C[M K]"
  (declare (type (simple-array single-float (*)) a b c)
	   (type (unsigned-byte 32) m n k a-offset b-offset c-offset)
	   (optimize (speed 3) (safety 0)))
  (dotimes (mi m)
    (dotimes (ni n)
      (dotimes (ki k)
	(incf (setf c (+ c-offset (* mi K) ni))
	      (* (aref a (+ a-offset (* mi n) ki))
		 (aref b (+ b-offset (* ki k) ni))))))))
```

So first, we need the device to execute the node. cl-waffe2 widely adapts CLOS anywhere, of course, all tensors are subclass of `AbstractTensor`. cl-waffe2 provides the `LispTensor` backend in standard which work on CPU. Matrix allocation etc. are already coded, so it is sufficient to create a new class by inheritance: `MyTensor` when you want to make an extension for CPU.

```lisp
(defclass MyTensor (LispTensor) nil)
```

`Matrix Multiplication` is the operation performed in every two dimensions. If the given tensor is 3D/4D..., it is batched and applied. If it is sliced, the offsets are added in that part. Regardless of which matmul implementation is used, but **at least** 2D tensor is needed, the process of adding offsets and batching follows the specification of cl-waffe2. So, we are going to use the [call-with-view](https://hikettei.github.io/cl-waffe2/generic-tensor/#function-call-with-view) function returning an S-expression that optimized and collapsed loops through a given tensor up to a defined dimension in order to maximize parallelization of kernel functions. And, describe the steps of calling `gemm!` as if writing a macro. (Of course, there is another way that writes a function directly though.)

One of the implementation of the device, can be given by the [define-impl](https://hikettei.github.io/cl-waffe2/nodes/#macro-define-impl) macro.


```lisp
(define-impl (MatmulNode-Revisit :device MyTensor)
	     :save-for-backward (t t nil)
	     :forward ((self a b c)
		       `(,@(call-with-view
			    #'(lambda (a-view b-view c-view)
			        ;; a-view = [a-view[dim=-2] a-view[dim=-1]]
				;; view contains: size, offset, stride
				`(gemm!
				  ,(size-of a-view 0) ;; (size-of view-list dim)
				  ,(size-of b-view 0)
				  ,(size-of c-view 1)
				  ,(offset-of a-view 0) (tensor-vec ,a)
				  ,(offset-of b-view 0) (tensor-vec ,b)
				  ,(offset-of c-view 0) (tensor-vec ,c)))
			    `(,a ,b ,c)
			    :at-least-dim 2)
			 ;; Returning C
			 ,c)))
```

A blueprint of the lambda functions described here is later compiled by `(compile nil body)`, and depending on the slices(offsets), ranks, dtypes, shapes, and permutations, functions are cached so users don't have to worry about the performance issue due to `eval`. On the contrary, the loop order is optimised (reordering, collapsing and lparallel) and can be expected to be about `1.1` times faster on average compared to loops without them.

So, we've got new implementation of `gemm`, let's get this going. Your `MyTensor` is registered as a cl-waffe2 device.

```lisp
CONCEPTS> (show-backends)

─────[All Backends Tree]──────────────────────────────────────────────────

[*]LISPTENSOR: Common Lisp implementation on matrix operations
    └[*]CPUTENSOR: OpenBLAS=available *simd-extension-p*=available
        └[*]MYTENSOR: No status.
        └[-]JITCPUTENSOR: compiler=gcc flags=(-fPIC -O3 -march=native) viz=NIL

[-]SCALARTENSOR: is a special tensor for representing scalar values.
    └[-]JITCPUSCALARTENSOR: Use with JITCPUTensor

([*] : in use, [-] : not in use.)
Add a current-backend-state method to display the status.
─────[*using-backend*]───────────────────────────────────────────────────

Priority: Higher <───────────────────────────>Lower
                  MYTENSOR CPUTENSOR LISPTENSOR 

(use with-devices macro or set-devices-toplevel function to change this parameter.)
NIL
```

(P.S.: Classes in an inheritance relationship in this tree are regarded as compatible with each other.)

There's two ways to declare `MyTensor` is a valid device that cl-waffe2 can use: locally using the `with-devices` macro, or using the `set-devices-toplevel` function. In this case, we use the `with-devices` macro and locally declares operations are performed under `MyTensor` backend.

```lisp
;; Gemm with Lisp
(defun test-gemm (&key (bench nil))
  (with-devices (MyTensor)
    (let ((a (randn `(100 100)))
	  (b (randn `(100 100)))
	  (c (make-input `(100 100) nil)))
      
      (proceed
       (call (MatmulNode-Revisit) a b c)
       :measure-time bench))))

;; Gemm with OpenBLAS
;; Set bench=t, and measures time
;; As excepted, this one is 20~30times faster.
(defun test-gemm-cpu (&key (bench nil))
  (with-devices (CPUTensor)
    (let ((a (randn `(100 100)))
	  (b (randn `(100 100)))
	  (c (make-input `(100 100) nil)))

      (proceed
       (call (MatmulNode :float) a b c)
       :measure-time bench))))
```

The moment AbstractNode is defined, the constructor function `(Node-Name args)` is also defined being called its forward propagation with the `call` or `forward` method. As with PyTorch and Chainer, `call` can be used to record the graph, but gemm cannot be executed at that time. You also have to tell a one more thing; "When will the results be needed?". The fundamental function is [build](https://hikettei.github.io/cl-waffe2/generic-tensor/#function-build), and the [proceed](https://hikettei.github.io/cl-waffe2/base-impl/#function-proceed) functions, which makes it differentiable, is easy to use in the REPL and debug.

```lisp
CONCEPTS> (test-gemm)

{MYTENSOR[float] :shape (100 100) :named :TENSOR 
  :vec-state [computed]
  ((-1.1147273    1.2072208     -0.25984392   ~ 0.038998634   -1.2577316    1.049665)                     
   (-0.28503713   0.3086878     -0.0664424    ~ 0.009971998   -0.3216035    0.2684006)   
                  ...
   (-0.19921443   0.21574406    -0.04643706   ~ 0.0069694985  -0.22477092   0.18758704)
   (0.9453751     -1.0238167    0.22036776    ~ -0.03307386   1.0666538     -0.89019716))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

Btw, the proceed functions provides more; for measuring execution time, and profiling your computation node.

```lisp
CONCEPTS> (proceed-time (!add 1 1))
[proceed-time] build ->
Evaluation took:
  0.000 seconds of real time
  0.000039 seconds of total run time (0.000038 user, 0.000001 system)
  100.00% CPU
  86,386 processor cycles
  0 bytes consed
  
[proceed-time] With allocation time:
Evaluation took:
  0.000 seconds of real time
  0.000030 seconds of total run time (0.000021 user, 0.000009 system)
  100.00% CPU
  65,918 processor cycles
  0 bytes consed
  
[proceed-time] Without allocation time:
Evaluation took:
  0.000 seconds of real time
  0.000009 seconds of total run time (0.000008 user, 0.000001 system)
  100.00% CPU
  17,032 processor cycles
  0 bytes consed

CONCEPTS> (proceed-bench (!softmax (randn `(100 100))))
[Sorted by Instructions]
 Time(s)  |   Instruction ( * - Beyonds the average execution time)
1.4e-5    | <WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID10689 <= op(TID10689{float, (100 100)} <Input>TID10601{float, (100 100)})>
2.0e-6    | <WfInst[op=VIEWTENSORNODE-T]         : TID10683 <= op(TID10641{float, (100 1)} TID10683{float, (100 1)})>
2.0e-6    | <WfInst[op=SCALARMUL-CPUTENSOR]      : TID10683 <= op(TID10683{float, (100 1)} <Input>TID10610{float, (1)})>
1.0e-6    | <WfInst[op=VIEWTENSORNODE-T]         : TID10683 <= op(TID10683{float, (100 100)} TID10683{float, (100 1)})>
8.4e-5*   | <WfInst[op=ADDNODE-CPUTENSOR]        : TID10683 <= op(TID10683{float, (100 100)} <Input>TID10601{float, (100 100)})>
1.0e-6    | <WfInst[op=VIEWTENSORNODE-T]         : TID10683 <= op(TID10683{float, (100 1)} TID10683{float, (100 100)})>
3.3e-5*   | <WfInst[op=SCALARDIV-CPUTENSOR]      : TID10683 <= op(TID10683{float, (100 1)} <Input>TID10605{float, (1)})>
2.0e-6    | <WfInst[op=VIEWTENSORNODE-T]         : TID10683 <= op(TID10683{float, (100 100)} TID10683{float, (100 1)})>
5.5e-5*   | <WfInst[op=SUBNODE-CPUTENSOR]        : TID10689 <= op(TID10689{float, (100 100)} TID10683{float, (100 100)})>
1.0e-5    | <WfInst[op=EXPNODE-CPUTENSOR]        : TID10689 <= op(TID10689{float, (100 100)} TID10689{float, (100 100)})>
2.0e-6    | <WfInst[op=SCALARMUL-CPUTENSOR]      : TID10683 <= op(TID10683{float, (100 1)} <Input>TID10738{float, (1)})>
1.0e-6    | <WfInst[op=VIEWTENSORNODE-T]         : TID10683 <= op(TID10683{float, (100 100)} TID10683{float, (100 1)})>
1.12e-4*  | <WfInst[op=ADDNODE-CPUTENSOR]        : TID10683 <= op(TID10683{float, (100 100)} TID10689{float, (100 100)})>
5.5e-5*   | <WfInst[op=DIVNODE-CPUTENSOR]        : TID10689 <= op(TID10689{float, (100 100)} TID10683{float, (100 100)})>

14 Instructions | 7 Tensors | Overheads due to SV4BW(...) -> 6.0e-6(s) 

 Total Time: 3.74e-4 sec

[Sorted by topK]
 Instruction                         | Total time (s) | Time/Total (n-sample=1)
<WfInst[op=ADDNODE-CPUTENSOR]        | 1.9600001e-4   | 52.406418%
<WfInst[op=SUBNODE-CPUTENSOR]        | 5.5e-5         | 14.705883%
<WfInst[op=DIVNODE-CPUTENSOR]        | 5.5e-5         | 14.705883%
<WfInst[op=SCALARDIV-CPUTENSOR]      | 3.3e-5         | 8.823529%
<WfInst[op=MOVETENSORNODE-CPUTENSOR] | 1.4e-5         | 3.7433155%
<WfInst[op=EXPNODE-CPUTENSOR]        | 1.0e-5         | 2.6737967%
<WfInst[op=VIEWTENSORNODE-T]         | 7.0e-6         | 1.8716577%
<WfInst[op=SCALARMUL-CPUTENSOR]      | 4.0e-6         | 1.0695187%
{MYTENSOR[float] :shape (100 100) :id TID10689 
  :vec-state [computed]
  ((0.01518923   0.049908508  0.017617546  ~ 0.012779288  0.0020210263 0.0018769704)                    
   (0.05841501   0.016887225  0.007463644  ~ 0.0027627628 0.0033548716 0.0028829984)   
                 ...
   (0.0016393916 0.0048142807 0.057292804  ~ 0.0036875184 0.015796835  0.0014674882)
   (0.03651239   0.0038913002 0.009065828  ~ 0.0060259635 0.012457987  0.003022789))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: DIVNODE-CPUTENSOR (A[~] B[~] -> A[~])>}
```

Lazy Evaluation Programming is not as hard as you'd think - Functions generated by AbstractNode can be embedded in your CommonLisp code in a natural way. As the simplest and basic example, the build function traces and compiles the network from the endpoints of the computation nodes, returning the `Compiled-Composite` class which can invoked its forward propagation by the `forward` method:

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

And `node->defun`, `node->lambda`. If you want to create a dedicated Tensor to store the calculation results, you should use the [make-input](https://hikettei.github.io/cl-waffe2/generic-tensor/#function-make-input) function with `name=nil`. InputTensor does not guarantee that the elements are filled with zeros, but the compiler can automatically reconnect them and reduce memory usage.

```lisp
(defun my-matmul (a b)
  (let* ((m (first  (shape a)))
	 (k (second  (shape b)))
	 (c (make-input `(,m ,k) nil))) ;; C as output trensor
    (call (MatmulNode-Revisit) a b c)))

(print
 (proceed (my-matmul (randn `(3 3)) (randn `(3 3)))))

;; Composing (!softmax (matmul a b))
(node->defun %mm-softmax (A[m n] B[n k] -> C[m k])
  (!softmax (my-matmul a b)))

;; Here's also (node->lambda (A[m n] B[n k] -> C[m k]) ...)

;; JIT Enabled Matrix Operations
(defun local-cached-matmul ()
  ;; Works like Lisp Function
  (print (time (%mm-softmax (randn `(3 3)) (randn `(3 3)))))
  (print (time (%mm-softmax (randn `(3 3)) (randn `(3 3))))))
```

And last, The `set-devices-toplevel` function make cl-waffe2 use `MyTensor` anywhere.

```lisp
(set-devices-toplevel 'MyTensor 'CPUTensor 'LispTensor)
```

## Advanced Network Constructions

We call a set of composed `AbstractNode`, or a chunk of nodes with trainable parameters, a [Composite](https://hikettei.github.io/cl-waffe2/nodes/#class-composite) defined by the [defmodel](https://hikettei.github.io/cl-waffe2/nodes/#macro-defmodel) macro.

```lisp
(defmodel (LayerNorm-Revisit (self normalized-shape &key (eps 1.0e-5) (affine T))
	   :slots ((alpha :initform nil :accessor alpha-of)
		   (beta  :initform nil :accessor beta-of)
		   (shape :initform nil :initarg :normalized-shape :accessor dim-of)
		   (eps   :initform nil :initarg :eps :accessor eps-of))
	   ;; Optional
	   :where (X[~ normalized-shape] -> out[~ normalized-shape])
	   :on-call-> layer-norm)

  ;; Constructor
  (when affine
    (setf (alpha-of self) (parameter (ax+b `(,@normalized-shape) 0 1))
	  (beta-of  self) (parameter (ax+b `(,@normalized-shape) 0 0)))))

(defmethod layer-norm ((self LayerNorm-Revisit) x)
  (with-slots ((alpha alpha) (beta beta)) self
    (let* ((last-dim (length (dim-of self)))
	   (u (!mean x :axis (- last-dim) :keepdims t))
	   (s (!mean (!expt (!sub x u) 2) :axis (- last-dim) :keepdims t))
	   (x (!div (!sub x u)
		    (!sqrt (!add (->contiguous s) (eps-of self))))))

     ;; !flexible = (%transform alpha[i] -> [~ i])
     ;; both inserts an broadcastable axis
      (if (and alpha beta)
	  (!add (!mul x (%transform alpha[i] -> [~ i]))
	        (!flexible beta)) 
	  x))))
```

As a chunk of nodes with trainable parameter, Composites can be used merely a subroutine:

```lisp
(proceed
    (call (LayerNorm-Revisit `(10)) (randn `(10 10 10))))
```

If you use Composite as a set of composed `AbstractNode`, they're compiled into another types:

```lisp
(defmodel (Softmax-Model (self)
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (let* ((x1 (!sub x (!mean x  :axis 1 :keepdims t)))
                              (z  (!sum   (!exp x1) :axis 1 :keepdims t)))
                         (!div (!exp x1) z)))))

(defmodel-as (Softmax-Model)
  :where (A[~] -> B[~])
  :asif :function :named %softmax)

(%softmax (randn `(10 10)))
```

## Composing

It is not elegant to use `call` more than once when composing multiple models.

```lisp
(call (AnyModel1)
      (call (AnyModel2)
             (call (AnyModel3) X)))
```

Instead, you can use the the [call->](https://hikettei.github.io/cl-waffe2/utils/#function-call-) function:

```lisp
(call-> X
        (AnyModel1)
        (AnyModel2)
        (AnyModel3))
```

If you wanted to insert a function to construct a computation node here, you can also use [asnode](https://hikettei.github.io/cl-waffe2/utils/#function-asnode) function to make the function recognised as a Composite.

```lisp
(call-> X
        (AnyModel1)
        (asnode #'!softmax)
        (asnode #'!view 0) ;; Slicing the tensor: (!view x 0 t ...)
        (asnode #'!add 1.0) ;; X += 1.0
        (asnode #'!matmul Y) ;; X <- Matmul(X, Y)
        )
```

If the Composite can be implemented using only `call->`, the [defsequence](https://hikettei.github.io/cl-waffe2/utils/#macro-defsequence) can be used for a short description:

```lisp
(defsequence MLP (in-features)
    "Docstring (optional)"
    (LinearLayer in-features 512)
    (asnode #'!tanh)
    (LinearLayer 512 256)
    (asnode #'!tanh)
    (LinearLayer 256 10))

;; Sequences can receive only a single argument.
(call (MLP 786) (randn `(10 786)))
```

## Make everything user-extensible

### Customized Autodiff - cl-waffe2 as a graph processing library

(TODO)

```lisp
(defclass MyScalarTensor (ScalarTensor) nil)
(set-devices-toplevel 'MyTensor 'CPUTensor 'LispTensor 'MyScalarTensor)
```

```lisp
(define-op (MyMul (self)
	    :where (A[scal] B[scal] -> A[scal] where scal = 1)
	    :out-scalar-p t
	    :save-for-backward-names (a b)
	    :forward ((self a b)
		      (with-setting-save4bw ((a a) (b b))
			(setf (tensor-vec a) (* (tensor-vec a) (tensor-vec b)))
			a))
	    :backward ((self dy)
		       (with-reading-save4bw ((a a) (b b))
			 (values
			  (make-tensor
			   (* (tensor-vec dy) (tensor-vec b)))
			  (make-tensor
			   (* (tensor-vec dy) (tensor-vec a))))))))

(define-op (MySin (self)
	    :where (A[scal] out[scal] -> out[scal] where scal = 1)
	    :out-scalar-p t
	    :save-for-backward-names (a)
	    :forward ((self a out)
		      (with-setting-save4bw ((a a))
			(setf (tensor-vec out) (sin (tensor-vec a)))
			out))
	    :backward ((self dy)
		       (with-reading-save4bw ((a a))
			 (values
			  (make-tensor
			   (* (tensor-vec dy)
			      (cos (tensor-vec a))))
			  nil)))))
(defun !mymul (a b)
  (call (MyMul) a b))

(defun !mysin (x)
  (call (MySin) x (make-clone x)))

(defun try-original-autodiff ()
  (let ((a (parameter (make-tensor 1))))
    (proceed-backward (!mysin (!mysin a)))
    (grad a)))
```

### Differentiable Programming

(TODO)

```lisp
(defoptimizer (MySGD (self param &key (lr 1e-3))
	       :slots ((lr :initarg :lr :reader sgd-lr))))

(node->defun %step-sgd (Param[~] Grad[~] Lr[scal] -> Param[~] where scal = 1)
  (A-=B param (A*=scal grad lr)))

(defmethod step-optimize ((optimizer MySGD))
  (let* ((lr    (make-tensor (sgd-lr optimizer)))
	 (param (read-parameter optimizer))
	 (grad  (grad param)))
    (with-no-grad
      (%step-sgd param grad lr))))
```

```lisp
(defun simple-opt-model ()
  (let* ((loss (!mean (!matmul (parameter (randn `(3 3)))
			       (parameter (randn `(3 3))))))
	 (model (build loss)))

    (mapc (hooker x (MySGD x :lr 1e-3)) (model-parameters model))
    
    (forward model)
    (backward model)
    
    (mapc #'call-optimizer! (model-parameters model))))
    
({MYTENSOR[float] :shape (3 3) -> :view (<T> <T>) -> :visible-shape (3 3)  
  ((0.25052267  -0.16212857 -1.3183842)
   (-1.078968   0.27860558  0.40701634)
   (-0.10987697 -1.2562615  0.6179133))
  :facet :exist
  :requires-grad T
  :optimizer <AbstractOptimizer: MYSGD() -> TID12604>}
 {MYTENSOR[float] :shape (3 3) -> :view (<T> <T>) -> :visible-shape (3 3)  
  ((-0.5223165  2.3579814   0.13172081)
   (0.57671905  0.56324756  1.1230979)
   (0.10274803  0.008530198 1.7588508))
  :facet :exist
  :requires-grad T
  :optimizer <AbstractOptimizer: MYSGD() -> TID12610>})
```

See also: [Examples](https://github.com/hikettei/cl-waffe2/tree/master/examples)

## Maximize the benefits of Graph-Level Optimization

(TODO)

```lisp
~~ [Steps] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.  Network Construction : Create a network of AbstractNode with multiple backends.
2.  Sorting/Pruning      : Sort the network and prune unused nodes.
3.  In-place mutation    : Optimize the list by deleting unused MoveTensorNode.
4.  More Localize        : Reconnecting InputTensors, the comiler optimizes the locality of memory.
5.  Reschedule           : Create an allocation planning considering 4. and in-place ops: !view !permute !reshape etc.
6.  Backward(Optional)   : Construct backward propagation
7.  Adjoint Optimization : Minimizes the number of copies arising at adjoints
8.  Compile/Inline       : If any, compiles lisp blueprints generated by call-with-view (If cached, ignored)
9.  Rewriting            : If any, replaces the list of declared patterns by the defpath macro
10. Completed -> Compiled-Composite is cached/inlined everywhere
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

(↓適当に書いたから絶対Grammarlyしたほうがいい)

The most challenging part of this framework is (8.), `call-with-view`. As far as I researched, Most of Previous works of JIT Compilers generates C/C++ codes and jit compiing them by invoking gcc but it means an unignorable overhead will occurs when compiling. cl-waffe2, however, compiles into Common Lisp Code, which is approximately 20x timers faster than invoking gcc, and with the appropriate optimization, it is not impossible to make it run as fast as C/C++. Plus, extensibility to other devices are also our major concerns. The call-with-view function maximizes the performance of calling foreign libraries; leave the less steps of iterations, the larger part of arrays, to foreign functions called via `cffi`. Compiled lambda fucntions by call-with-view are cached and dispatched depending on their dtypes, shapes, views... So, cl-waffe2 compiler performs signifcantly faster compiling time.

InputTensor is ...

Plus, Common Lisp macro systems are enabled pseudo-AoT Shape-Error detecting ...

## Interop: Common Lisp Array and cl-waffe2 AbstractTensor

(TODO)

See also: [Converting AbstractTensor and other arrays](https://hikettei.github.io/cl-waffe2/utils/#tensor-facet-converting-abstracttensor-anything)

## Debugging

(TODO)

cl-waffe2 is enough clever to detect Shape-Error and suggest an alternative arising from wrong inputs. In this case, both ranks are invaild because broadcasing rank-up rule is not applied in cl-waffe2:

```lisp
(!add (randn `(3 3)) (randn `(3)))
```

If you do this, you will get the following error before **running the operation**

```lisp
[Shaping Error]: The AbstractNode ADDNODE-CPUTENSOR was called with invaild arguments.

 The constraint:
    ADDNODE-CPUTENSOR: (A[~] B[~] -> A[~])

Received:
    (forward
        (ADDNODE-CPUTENSOR ...)
         CPUTENSOR{FLOAT}(3 3)
         CPUTENSOR{FLOAT}(3) ─ B: The length of ~~ do not match. The Rank is too low 
        )

B:
    ─ the 1th shape is (3) but it violates ~ = (3 3)
    ─ The given rank 2 do not match declared: (3)

Excepted:
    (forward
        (ADDNODE-CPUTENSOR ...)
        A(3 3)
        B(3) ─> B: 
        )

B:
    ─ 
    ─ Use (!flexible tensor) to explict a rank-up rule of broadcasting.

Predicted outputs of ADDNODE-CPUTENSOR:  ((3 3))

The operation was:
<Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>
```

When adding or repeating ranks by broadcasting rule, it is necessary to declare in advance in which position they are to be added:

```lisp
;; X[a b c] -> X[~ a b c]
(!flexible x :at 0)

;; X[a] -> X[~ a]
(%transform (randn `(3))[i] -> [~ i])
```

Following the suggestion, fix the code:

```lisp
(defparmeter out (!add (randn `(3 3)) (!flexible (randn `(3)))))
```

And passed:

```lisp
(proceed out)
```

This is a case of before execution, speaking of runtime error (e.g.: floating-point overflow), it gets a bit complicated; you have to face the disassembled code to find out the details.

When doing (!sqrt x) where x is a negative number:

```lisp
(proceed
 (!sin
  (!sqrt
   (!mul -1.0
         (!sin (ax+b `(3 3) 0 1))))))
```

As excepted it produces FLOATING-POINT-INVAILD-OPERATION:

```lisp
cl-waffe2 VM: Encountered Runtime Error at 3th instruction.
disassemble:
0 : <WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID14200 <= op(TID14200{float, (3 3)} <Input>TID14197{float, (3 3)})>
1 : <WfInst[op=SINNODE-CPUTENSOR]        : TID14200 <= op(<Input>TID14197{float, (3 3)} TID14200{float, (3 3)})>
2 : <WfInst[op=SCALARMUL-CPUTENSOR]      : TID14200 <= op(TID14200{float, (3 3)} <Input>TID14218{float, (1)})>
3*: <WfInst[op=SQRTNODE-CPUTENSOR]       : TID14200 <= op(TID14200{float, (3 3)} TID14200{float, (3 3)})>
4 : <WfInst[op=SINNODE-CPUTENSOR]        : TID14200 <= op(TID14200{float, (3 3)} TID14200{float, (3 3)})>


condition:
  arithmetic error FLOATING-POINT-INVALID-OPERATION signalled
```

Use the [disassemble-waffe2-ir](https://hikettei.github.io/cl-waffe2/vm/#function-disassemble-waffe2-ir) function to check the full disassembled code instead of proceed:

```lisp
(disassemble-waffe2-ir
 (!sin
  (!sqrt
   (!mul -1.0
         (!sin (parameter (ax+b `(3 3) 0 1)))))))
```

```lisp
disassemble-waffe2-ir:
 [Forward]: 
<WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID14654 <= op(TID14654{float, (3 3)} <Param>TID14649{float, (3 3)})>
<WfInst[op=SINNODE-CPUTENSOR]        : TID14654 <= op(<Param>SV4BW(TID14649{float, (3 3)}) TID14654{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR]      : TID14654 <= op(SV4BW(TID14654{float, (3 3)}) <Input>TID14675{float, (1)})>
<WfInst[op=SQRTNODE-CPUTENSOR]       : TID14654 <= op(SV4BW(TID14654{float, (3 3)}) TID14654{float, (3 3)})>
<WfInst[op=SINNODE-CPUTENSOR]        : TID14654 <= op(SV4BW(TID14654{float, (3 3)}) TID14654{float, (3 3)})>

5 Instructions | 2 Tensors | 1 Scalars


 [Pullback]: 
<WfInst[op=MOVETENSORNODE-CPUTENSOR]    : TID14764 <= op(TID14764{float, (3 3)} <Input>TID14742{float, (3 3)})>
<WfInst[op=COSNODE-CPUTENSOR]           : TID14732 <= op(TID14732{float, (3 3)} TID14732{float, (3 3)})>
<WfInst[op=MULNODE-CPUTENSOR]           : TID14764 <= op(TID14764{float, (3 3)} TID14732{float, (3 3)})>
<WfInst[op=INVERSETENSORNODE-CPUTENSOR] : TID14710 <= op(TID14710{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR]         : TID14710 <= op(TID14710{float, (3 3)} <Input>TID14782{float, (1)})>
<WfInst[op=MULNODE-CPUTENSOR]           : TID14764 <= op(TID14764{float, (3 3)} TID14710{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR]         : TID14764 <= op(TID14764{float, (3 3)} <Input>TID14675{float, (1)})>
<WfInst[op=COSNODE-CPUTENSOR]           : TID14665 <= op(TID14665{float, (3 3)} TID14665{float, (3 3)})>
<WfInst[op=MULNODE-CPUTENSOR]           : TID14764 <= op(TID14764{float, (3 3)} TID14665{float, (3 3)})>
<WfInst[op=MOVETENSORNODE-CPUTENSOR]    : <Input>TID14651 <= op(<Input>TID14651{float, (3 3)} TID14764{float, (3 3)})>

10 Instructions | 6 Tensors | 2 Scalars
```


## Graph Rewriting

(Still Experimental, but coming soon...)

We gonna talk about `defpath` which enables theano-like symbolic differentiation and device-specific optimizations.


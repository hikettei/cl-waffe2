
(in-package :cl-waffe2.docs)

;; node to apis ha wakeru?

(with-page *vm* "cl-waffe2 VM"
  (insert "
The package `cl-waffe2/vm` is the central of system, and features are focused on low-level stuffs: compiling/optimizing/rewriting cl-waffe2 IRs and how they're executed. So, most of APIs are accesible by convinient API wrappers of other packages.

- Global Variables
    - [optimization level](./#parameter-opt-level)
    - [logging](./#parameter-logging-vm-execution)
- IR and Compiler
    - [WfInstruction](./#struct-wfinstruction)
    - [compiler](./#function-compile-forward-and-backward)
    - [acceptor](./#function-accept-instructions)
- Analyzing compiled codes
    - [disassemble](#function-disassemble-waffe2-ir)
    - [profiling](#function-benchmark-accept-instructions)
")
  (macrolet ((with-op-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))
    
    (with-op-doc '*opt-level* 'variable)
    (with-op-doc '*logging-vm-execution* 'variable)

    (with-op-doc 'WfInstruction 'structure)
    
    (with-op-doc #'cl-waffe2/vm:compile-forward-and-backward 'function)
    (with-op-doc #'cl-waffe2/vm:accept-instructions 'function)
    
    (with-op-doc #'cl-waffe2/vm:disassemble-waffe2-ir 'function
      (with-example
	"(with-output-to-string (out)
    (disassemble-waffe2-ir (!softmax (parameter (randn `(3 3))) :avoid-overflow nil) :stream out))"))
    
    (with-op-doc #'cl-waffe2/vm:benchmark-accept-instructions 'function
      (with-example
	"(with-output-to-string (out)
    (proceed-bench (!softmax (randn `(100 100))) :n-sample 100 :stream out))"))))

(with-page *base-impl-nodes* "Standard Nodes"
  (macrolet ((nodedoc (name)
	       `(insert "~a" ,(documentation (find-class name) 't)))
	     (caller-doc (name)
	       `(insert "~a" ,(documentation name 'function))))

    (nodedoc AddNode)
    (nodedoc SubNode)
    (nodedoc MulNode)
    (nodedoc DivNode)

    (nodedoc MoveTensorNode)

    (nodedoc AbsNode)
    (nodedoc Scalar-AbsNode)

    (nodedoc SignNode)
    (nodedoc Scalar-SignNode)
    
    (nodedoc SqrtNode)
    (nodedoc Scalar-SqrtNode)

    (nodedoc SquareNode)
    (nodedoc Scalar-SquareNode)

    (nodedoc SinNode)
    (nodedoc Scalar-SinNode)

    (nodedoc CosNode)
    (nodedoc Scalar-CosNode)

    (nodedoc TanNode)
    (nodedoc Scalar-TanNode)

    (nodedoc ASinNode)
    (nodedoc Scalar-ASinNode)

    (nodedoc ACosNode)
    (nodedoc Scalar-ACosNode)

    (nodedoc ATanNode)
    (nodedoc Scalar-ATanNode)

    (nodedoc SinHNode)
    (nodedoc Scalar-SinHNode)

    (nodedoc CosHNode)
    (nodedoc Scalar-CosHNode)

    (nodedoc TanHNode)
    (nodedoc Scalar-TanHNode)

    (nodedoc ASinHNode)
    (nodedoc Scalar-ASinHNode)

    (nodedoc ACosHNode)
    (nodedoc Scalar-ACosHNode)

    (nodedoc ATanHNode)
    (nodedoc Scalar-ATanHNode)

    (nodedoc ExpNode)
    (nodedoc Scalar-ExpNode)

    (nodedoc Log2Node)
    (nodedoc Scalar-Log2Node)

    (nodedoc Log10Node)
    (nodedoc Scalar-Log10Node)

    (nodedoc LogENode)
    (nodedoc Scalar-LogENode)

    (nodedoc Log1PNode)
    (nodedoc Scalar-Log1PNode)

    ;; TODO: pow/expt

    (nodedoc cl-waffe2/base-impl::LazyTransposeNode)

    (nodedoc ArgMax-Node)
    (nodedoc ArgMin-Node)

    (nodedoc MaxValue-Node)
    (nodedoc MinValue-Node)

    (nodedoc MatmulNode)

    (nodedoc Lazy-Function-Node)
    (nodedoc Lazy-Reduce-Node)

    (nodedoc Where-Operation-Node)
    (nodedoc Compare-Operation-Node)

    (nodedoc Im2ColNode)
    (nodedoc Col2ImNode)
    ))


(with-page *base-impl* "Basic APIs"
  (macrolet ((nodedoc (name)
	       `(insert "~a" ,(documentation (find-class name) 't)))
	     (caller-doc (name)
	       `(insert "~a" ,(documentation name 'function))))

    (caller-doc !matrix-add)
    (caller-doc !matrix-sub)
    (caller-doc !matrix-mul)
    (caller-doc !matrix-div)
    
    (caller-doc !reciprocal)
    (caller-doc !scalar-add)
    (caller-doc !scalar-sub)
    (caller-doc !scalar-mul)
    (caller-doc !scalar-div)
    
    (caller-doc !sas-add)
    (caller-doc !sas-sub)
    (caller-doc !sas-mul)
    (caller-doc !sas-div)
    
    (caller-doc !add)
    (caller-doc !sub)
    (caller-doc !mul)
    (caller-doc !div)

    (caller-doc !+)
    (caller-doc !-)
    (caller-doc !*)
    (caller-doc !/)

    (caller-doc !move)
    (caller-doc !copy)

    (caller-doc !permute)
    (caller-doc !reshape)
    (caller-doc !view)
    ;; unsqueeze/squeeze
    (caller-doc !flatten)
    (caller-doc !rankup)
    (caller-doc ->scal)
    (caller-doc ->mat)
    (caller-doc ->contiguous)
    
    (caller-doc proceed)
    (caller-doc proceed-time)
    (caller-doc proceed-backward)
    (caller-doc proceed-bench)

    (caller-doc %transform)
    (caller-doc !flexible)

    (caller-doc !abs)
    (caller-doc !sign)
    (caller-doc !sqrt)
    (caller-doc !square)
    (caller-doc !sin)
    (caller-doc !cos)
    (caller-doc !tan)
    (caller-doc !asin)
    (caller-doc !acos)
    (caller-doc !atan)
    (caller-doc !sinh)
    (caller-doc !cosh)
    (caller-doc !tanh)
    (caller-doc !asinh)
    (caller-doc !acosh)
    (caller-doc !atanh)

    (caller-doc !exp)
    (caller-doc !log2)
    (caller-doc !log10)
    (caller-doc !logE)
    (caller-doc !log1p)

    (caller-doc !expt)

    (caller-doc !sum)
    (caller-doc !mean)

    (caller-doc !argmax)
    (caller-doc !argmin)

    (caller-doc !max)
    (caller-doc !min)

    (caller-doc !t)
    (caller-doc !matmul)
    (caller-doc !dot)

    (caller-doc lazy)
    (caller-doc lazy-reduce)

    (caller-doc !where)
    (caller-doc !compare)


    (caller-doc A>scal)
    (caller-doc A<scal)
    (caller-doc A>=scal)
    (caller-doc A<=scal)
    (caller-doc A=scal)

    (caller-doc A>B)
    (caller-doc A<B)
    (caller-doc A>=B)
    (caller-doc A<=B)
    (caller-doc A=B)

    (caller-doc padding)
    (caller-doc broadcast-to)
    ))


(with-page *cl-waffe2-package* "[package] cl-waffe2"
  (insert "The `cl-waffe2` package provides utilities for a wide range needs: Object Convertion, Advance Network Construction, Trainer, and so on.")

  (macrolet ((with-op-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))

    (with-section "[Tensor Facet] Converting AbstractTensor <-> Anything"
      (insert "If you're looking for the way to create an AbstractTensor from a Common Lisp array or manipulate an AbstractTensor as a Common Lisp array, this section is perfect for you. Here we provide a common APIs for the conversion between AbstractTensor and other matrix types. The most basic method is a `convert-tensor-facet` method and we're welcome to add a new method by users. Other functions are macros work by assigning a method according to the type of object and the direction. Plus, conversions are performed while sharing pointers as much as possible. If turned out to be they can't be shared, the with-facet macro forces a copy to be performed and pseudo-synchronises them.~%")
      
      (with-op-doc #'convert-tensor-facet 't)
      (with-op-doc #'change-facet 't)
      (with-op-doc #'device-as 't)
      (with-op-doc #'->tensor 't)
      (with-op-doc (macro-function 'with-facet) 't)
      (with-op-doc (macro-function 'with-facets) 't))

    (with-section "Advanced Network Construction"
      (insert "Powerful macros in Common Lisp enabled me to provide an advanced APIs for make the construction of nodes more systematic, and elegant. Computational nodes that are lazy evaluated can be treated as pseudo-models, for example, even if they are created via functions. And, APIs in this section will make it easy to compose/compile several nodes.~%")
      (with-op-doc #'asnode 't)
      (with-op-doc (macro-function 'RepeatN) 't)
      (with-op-doc #'call-> 't)
      (with-op-doc (macro-function 'defsequence) 't)
      (with-op-doc (macro-function 'hooker) 't)
      (with-op-doc (macro-function 'node->lambda) 't)
      (with-op-doc (macro-function 'node->defun) 't)
      (with-op-doc (macro-function 'node->defnode) 't)
      )
    
    (with-op-doc #'show-backends 'function)
    (with-op-doc #'set-devices-toplevel 'function)))

(with-page *lisp-tensor-backend* "[backend] :cl-waffe2/backends.lisp"
  (insert
   "
The package `:cl-waffe2/backends.lisp` provides a `LispTensor` backend which is designed with the aim of portalibity, not performance. All matrix operations are performed via kernels written in ANSI Common Lisp.

In order to use this backend, add this line:

```lisp
(with-devices (LispTensor)
   ;; body
   )
```
")

  (macrolet ((with-op-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))
    (with-op-doc (find-class 'LispTensor) 't)))

(with-page *cpu-tensor-backend* "[backend] :cl-waffe2/backends.cpu"
  (insert
   "The package `:cl-waffe2/backends.cpu` provides a `CPUTensor` backend which relies most of kernel implementations on foreign libraries invoked via CFFI. (e.g.: OpenBLAS, oneDNN in the coming future).")

  (insert "
## Enabling the SIMD Extension

```sh
$ make build_simd_extension
```

See also: [cl-waffe2-simd](https://github.com/hikettei/cl-waffe2/tree/master/cl-waffe2-simd)

To get further performance on CPU, SIMD Extension must be installed on your device. This extension provides further SIMD-enabled CPUTensor operations (e.g.: !max/!min, Sparse Matrix Supports, vectorized mathematical functions of SLEEF, etc...). To use it, run `make build_simd_extension` in the same directory as cl-waffe2.asd. You can confirm that it works properly with the `(cl-waffe2:show-backends)` function.
")
  
  (macrolet ((with-op-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))
    (with-op-doc (find-class 'CPUTensor) 't)))


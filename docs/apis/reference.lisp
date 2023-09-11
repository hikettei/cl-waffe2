
(in-package :cl-waffe2.docs)

;; node to apis ha wakeru?

(with-page *vm* "cl-waffe2 VM"
  (macrolet ((with-op-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))
    (with-op-doc #'cl-waffe2/vm:disassemble-waffe2-ir 'function)
    (with-op-doc #'cl-waffe2/vm:benchmark-accept-instructions 'function)
    (with-op-doc #'cl-waffe2/vm:compile-forward-and-backward 'function)
    (with-op-doc #'cl-waffe2/vm:accept-instructions 'function)))

(with-page *base-impl-nodes* "Standard Nodes"
  (macrolet ((nodedoc (name)
	       `(insert "~a" ,(documentation (find-class name) 't)))
	     (caller-doc (name)
	       `(insert "~a" ,(documentation name 'function))))

    (nodedoc AddNode)
    (nodedoc SubNode)
    (nodedoc MulNode)
    (nodedoc DivNode)
    
    (nodedoc InverseTensorNode)
    
    (nodedoc ScalarAdd)
    (nodedoc ScalarSub)
    (nodedoc ScalarMul)
    (nodedoc ScalarDiv)

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

    ;; TODO: pow/expt

    (nodedoc cl-waffe2/base-impl::LazyTransposeNode)

    (nodedoc ArgMax-Node)
    (nodedoc ArgMin-Node)

    (nodedoc MaxValue-Node)
    (nodedoc MinValue-Node)

    (nodedoc MatmulNode)

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
    
    (caller-doc !inverse)
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
    (caller-doc !copy-force)

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

    (caller-doc !sum)
    (caller-doc !mean)

    (caller-doc !argmax)
    (caller-doc !argmin)

    (caller-doc !max)
    (caller-doc !min)

    (caller-doc !t)
    (caller-doc !matmul)
    (caller-doc !dot)

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
      (with-op-doc #'->tensor 't)
      (with-op-doc (macro-function 'with-facet) 't)
      (with-op-doc (macro-function 'with-facets) 't))

    (with-section "Advanced Network Construction"
      (insert "Powerful macros in Common Lisp enabled me to provide an advanced APIs for make the construction of nodes more systematic, and elegant. Computational nodes that are lazy evaluated can be treated as pseudo-models, for example, even if they are created via functions. And, APIs in this section will make it easy to compose/compile several nodes.~%")
      (with-op-doc (macro-function 'asnode) 't)
      (with-op-doc (macro-function 'call->) 't)
      (with-op-doc (macro-function 'defsequence) 't))

    (with-section "Trainer"
      (insert "(TODO)

```lisp
minimize!:
  ...


set-input:
  describe ...

predict:
  describe ..
```"))

    (with-op-doc #'show-backends 'function)
    (with-op-doc #'set-devices-toplevel 'function)

    ))

(with-page *lisp-tensor-backend* "[package] :cl-waffe2/backends.lisp"
  (insert
   "The package `:cl-waffe2/backends.lisp` provides an AbstractTensor `LispTensor` as an external backend, and designed with the aim of portalibity, not performance. Therefore, most implementations of this follow ANSI Common Lisp, so it will work in any environment but concerns remain about speed.

It is recommended that `LispTensor` are installed in the lowest priority of `*using-backend*`, and `Couldnt find any implementation for ...` error will never occurs.")

  (macrolet ((with-op-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))
    (with-op-doc (find-class 'LispTensor) 't)
    ))

(with-page *cpu-tensor-backend* "[package] :cl-waffe2/backends.cpu"
  (insert
   "The package `:cl-waffe2/backends.cpu` provides an AbstractTensor `CPUTensor` where most of its implementation relies on foreign libraries (e.g.: OpenBLAS, oneDNN in the coming future).")

  (insert "
## Enabling the SIMD Extension

For some instructions (e.g.: `!max` `!min`, sparse matrix supports, `SLEEF`, etc...), packages that provide SIMD-enabled CPUTensor implementations are not enabled by default as a design. To enable it, run `make build_simd_extension` in the same directory as cl-waffe2.asd. You can check that it is loaded properly with the `(show-backends)` function.
")
  
  (macrolet ((with-op-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))
    (with-op-doc (find-class 'CPUTensor) 't)
    ))

(with-page *cpu-jit-tensor-backend* "[package] :cl-waffe2/backends.jit.cpu"
  (insert "The package `:cl-waffe2/backends.jit.cpu` provides an AbstractTensor `JITCPUTensor` which accelerated by JIT Compiling to C code dynamically, (so this backend will require `gcc` as an additional requirement.)")
  (macrolet ((with-op-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))
    (with-op-doc '*default-c-compiler* 'variable)
    (with-op-doc '*compiler-flags* 'variable)
    (with-op-doc '*viz-compiled-code* 'variable)

    (with-op-doc (find-class 'JITCPUTensor) 't)
    (with-op-doc (find-class 'JITCPUScalarTensor) 't)

    (with-op-doc #'enable-cpu-jit-toplevel 'function)
    (with-op-doc (macro-function 'with-cpu-jit) 'function)
    
    ))


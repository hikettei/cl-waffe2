
(in-package :cl-waffe2.docs)

(with-page *generic-tensor* "AbstractTensor"
  (macrolet ((with-doc (name type &body body)
	       `(progn;;with-section (format nil "~(~a~)" (symbol-name ,name))
		  (placedoc ,name ,type)
		  ,@body)))

    ;; AbstractTensor
    (with-section "[class] AbstractTensor"
      (insert "
[class] `AbstractTensor`

~a"
	      (documentation (find-class 'AbstractTensor) 't)))

    (with-section "[function] tensor-vec"
      (insert "~%`(tensor-vec tensor)`

Reading the `vec` of tensor.  Not until tensor-vec is called, the new area isn't allocated."))
    
    (with-section "[function] mref"
      (insert "~%`(mref tensor &rest subscripts)`~%~%~a" (documentation 'mref 'function)))

    (with-section "[generic] vref"
      (insert "~%`(vref tensor index)`~%~%~a"
	      (documentation (car (c2mop:generic-function-methods #'vref)) 't)))
    
    (with-section "An form of tensors in cl-waffe2"
      (insert "There's a two type of tensors in cl-waffe2: `InputTensor` and `ExistTensor`, each state is called `facet` and the keyword `:input` `:exist` is dispatched respectively.~%")

      (insert "### ExistTensor
`ExistTensor` means a tensor with its vec **allocated** in the memory, that is, the same tensor as tensors you got when create a new tensor in `Numpy`, `PyTorch` or something.

`ExistTensor` can be created by the function `make-tensor`.

~%")
      (insert "### InputTensor
On the other hand, `InputTensor` is a tensor with its vec **unallocated** in the memory, in other words, this can be a `Lazy-Evaluated Tensor`.

`InputTensor` is created by the function `make-input`, and its shape can include a symbol.

In the network, `InputTensor` plays a role in being caches in the operation, or being a tensor that one may want to change its content later. (e.g.: training data).
~%"))
    
    (progn
      ;; Making a new tensor.
      (with-doc 'make-tensor 'function
	(with-example
	  "(make-tensor `(10 10) :initial-element 1.0)"))

      ;; with-example
      (with-doc 'make-input 'function
	(with-example
	  "(make-input `(a 10) :train-x)")
	(insert "The InputTensor named with a keyword is called `not-embodied tensor`, and can be changed its `vec` with `embody-input`")))

    ;; Composite
    
    (insert "~a"
	    (documentation (find-class 'Compiled-Composite) 't))
    ;; with-example      
    (with-doc 'build 'function
      (with-examples
	"(setq out (!add (randn `(10 10)) (make-input `(a 10) :X)))"
	"(multiple-value-list (build out))"))

    
    (insert "~a"
	    (documentation (car (c2mop:generic-function-methods #'set-input)) 't))

    (insert "~a"
	    (documentation (car (c2mop:generic-function-methods #'get-input)) 't))

    
    ;; Gradient Utils
    (with-section "[parameter] `*no-grad*`"
      (insert "[parameter] `*no-grad*`

~a" (documentation '*no-grad* 'variable)))

    
    (with-doc 'with-no-grad 'macro)
    
    (with-doc 'parameter 'function)

    (with-doc 'call-with-view 'function)

    (with-doc 'with-ranked-loop 'macro)

    (with-doc 'stride-of 'function)

    (with-doc 'size-of 'function)

    (with-doc 'offset-of 'function)

    (with-doc 'shape-equal 'function)
    
    (with-doc 'force-list 'function)

    (with-section "Compiling Options"
      (insert "TODO"))

    (with-section "Dtypes"
      (insert "TODO"))
    
    ))



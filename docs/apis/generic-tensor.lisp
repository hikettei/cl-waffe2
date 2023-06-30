
(in-package :cl-waffe2.docs)

(with-page *generic-tensor* "AbstractTensor"
  (macrolet ((with-doc (name type &body body)
	       `(with-section (format nil "~(~a~)" (symbol-name ,name))
		  (placedoc ,name ,type)
		  ,@body)))

    ;; AbstractTensor
    (with-section "AbstractTensor"
      (insert "
[class] `AbstractTensor`

~a"
	      (documentation (find-class 'AbstractTensor) 't))

      (with-doc 'make-tensor 'function
	(with-example
	  "(make-tensor `(10 10) :initial-element 1.0)"))

      ;; with-example
      (with-doc 'make-input 'function
	(with-example
	  "(make-input `(a 10) :train-x)")
	(insert "The InputTensor named with a keyword is called `not-embodied tensor`, and can be changed its `vec` with `embody-input`"))
      
      ;; with-example

      #|
      (with-doc 'embody-input 'function
	(with-examples
	  "(setq out (!add (randn `(10 10)) (make-input `(a 10) :x)))"
	  "(with-build (fw bw vars params) out
            (embody-input vars :x (randn `(10 10))) ;; :X = (randn `(10 10))
            (funcall fw))"))
      |#
      
      ;; with-example      
      (with-doc 'build 'function
	(with-examples
	  "(setq out (!add (randn `(10 10)) (make-input `(a 10) :X)))"
	  "(multiple-value-list (build out))"))

      
      ;; Accessors
      (with-section "tensor-vec"
	(insert "~%`(tensor-vec tensor)`

Accessing the pointer/array the tensor has. Not until tensor-vec is called, the new area isn't allocated."))

      (with-section "mref"
	(insert "~%`(mref tensor &rest subscripts)`~%~%~a" (documentation 'mref 'function)))

      (with-section "vref"
	(insert "~%`(vref tensor index)`~%~%~a"
		(documentation (car (c2mop:generic-function-methods #'vref)) 't))))

    (with-doc 'set-save-for-backward 'function)
    (with-doc 'read-save-for-backward 'function)

    
    ;; Gradient Utils
    (with-section "`*no-grad*`"
      (insert "[parameter] `*no-grad*`

~a" (documentation '*no-grad* 'variable)))

    
    (with-doc 'with-no-grad 'macro)
    
    (with-doc 'parameter 'function)

    (with-doc 'dtype->lisp-type 'function)

    (with-doc 'call-with-view 'function)

    (with-doc 'stride-of 'function)

    (with-doc 'size-of 'function)

    (with-doc 'offset-of 'function)

    (with-doc 'shape-equal 'function)

    (with-doc 'force-list 'function)
    ))



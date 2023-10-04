
(in-package :cl-waffe2.docs)

(with-page *generic-tensor* "AbstractTensor"
  (macrolet ((with-doc (name type &body body)
	       `(progn;;with-section (format nil "~(~a~)" (symbol-name ,name))
		  (placedoc ,name ,type)
		  ,@body)))

    (with-section "Working with AbstractTensor")
    
    (insert "~a" (documentation (find-class 'AbstractTensor) 't))
    (insert "~a" (documentation 'hook-optimizer! 'function))
    (insert "~a" (documentation 'call-optimizer! 'function))
    (insert "~a" (documentation 'reset-grad! 'function))
    
    (insert (documentation 'tensor-vec 'function))
    
    
    (progn
      (with-doc 'make-tensor 'function
	(with-example
	  "(make-tensor `(10 10) :initial-element 1.0)"))

      (with-doc 'make-input 'function
	(with-example
	  "(make-input `(a 10) :train-x)")))

    (with-section "Manipulating Gradients")

    (insert "~a" (documentation '*no-grad* 'variable))
    (with-doc 'with-no-grad 'macro)
    (with-doc 'parameter 'function)
    
    (with-section "Building functions from AbstractTensor")

    ;; Composite
    (insert "~a"  (documentation (find-class 'Compiled-Composite) 't))

    ;; with-example      
    (with-doc 'build 'function
      (with-examples
	"(setq out (!add (make-input `(a 10) :X) (make-input `(a 10) :Y)))"
	"(with-no-grad (build out :inputs `(:X :Y)))"))

    
    (insert "~a"
	    (documentation (car (c2mop:generic-function-methods #'set-input)) 't))

    (insert "~a"
	    (documentation (car (c2mop:generic-function-methods #'get-input)) 't))


    (with-section "Optimized and Ranked Tensor Iterators")

    (with-doc 'call-with-view 'function)
    (with-doc 'do-compiled-loop 'macro)

    (with-section "Save/Restore Weights")

    (with-doc 'State-Dict 'structure)
    (insert "~a" (documentation (macro-function 'define-model-format) 'function))

    (with-doc 'save-weights 'function)
    (with-doc 'load-weights 'function)
    (with-doc 'load-from-state-dict 'function)
    ))




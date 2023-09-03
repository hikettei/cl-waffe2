
(in-package :cl-waffe2.docs)

(with-page *generic-tensor* "AbstractTensor"
  (macrolet ((with-doc (name type &body body)
	       `(progn;;with-section (format nil "~(~a~)" (symbol-name ,name))
		  (placedoc ,name ,type)
		  ,@body)))

    (with-section "Working with AbstractTensor")
    
    (insert (documentation (find-class 'AbstractTensor) 't))
    (insert (documentation 'hook-optimizer!) 'function)
    (insert (documentation 'call-optimizer!) 'function)
    (insert (documentation 'reset-grad!) 'function)
    
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
	"(build out :inputs `(:X :Y))"))

    
    (insert "~a"
	    (documentation (car (c2mop:generic-function-methods #'set-input)) 't))

    (insert "~a"
	    (documentation (car (c2mop:generic-function-methods #'get-input)) 't))


    (with-section "Creating a ranked function with computing views")

    (with-doc 'call-with-view 'function)

    (with-doc 'with-ranked-loop 'macro)))



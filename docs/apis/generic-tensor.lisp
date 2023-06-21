
(in-package :cl-waffe2.docs)

(with-page *generic-tensor* "AbstractTensor"
  (macrolet ((with-doc (name type &body body)
	       `(with-section (format nil "~(~a~)" (symbol-name ,name))
		  (placedoc ,name ,type)
		  ,@body)))

    (with-section "AbstractTensor"
      (insert "
[class] AbstractTensor

~a"
	      (documentation (find-class 'AbstractTensor) 't))

      (with-section "tensor-vec"
	(insert "~%`(tensor-vec tensor)`

Accessing the pointer/array the tensor has. Not until tensor-vec is called, the new area isn't allocated."))

      (with-section "mref"
	(insert "~%`(mref tensor &rest subscripts)`~%~%~a" (documentation 'mref 'function)))

      (with-section "vref"
	(insert "~%`(vref tensor index)`~%~%~a"
		(documentation (car (c2mop:generic-function-methods #'vref)) 't))))

    (with-doc 'parameter 'function)
    (with-section "`*no-grad*`"
      (insert "[parameter] `*no-grad*`

~a" (documentation '*no-grad* 'variable)))
    
    (with-doc 'with-no-grad 'macro)
    (with-doc 'make-tensor 'function)
    (with-doc 'make-input 'function)

    (insert "(TODO) -> View APIs etc...")
    ;; TO Add: ViewInstruction
    ;; stride-of etc....

    ))



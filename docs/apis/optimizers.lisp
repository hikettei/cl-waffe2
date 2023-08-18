
(in-package :cl-waffe2.docs)

(with-page *optimizer* "cl-waffe2/optimizers"
  (macrolet ((with-op-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))

    (with-op-doc (find-class 'AbstractOptimizer) 't)
    (with-op-doc (macro-function 'defoptimizer) 'function)
    (with-op-doc (find-class 'SGD) 't)
    (with-op-doc (find-class 'Adam) 't)
    ))


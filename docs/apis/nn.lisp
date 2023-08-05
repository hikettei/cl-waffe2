
(in-package :cl-waffe2.docs)

(with-page *nn* "cl-waffe2/nn"
  (macrolet ((with-nn-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))
    (with-nn-doc '!relu 'function
      (with-example
	"(proceed (!relu (randn `(10 10))))"))
    (with-nn-doc '!gelu 'function)
    (with-nn-doc '!sigmoid 'function
      (with-example
	"(proceed (!sigmoid (randn `(10 10))))"))

    (with-nn-doc 'L1Norm 'function
      (with-example
	"(proceed (L1Norm (randn `(10 10)) (randn `(10 10))))"))
    (with-nn-doc 'mse 'function
      (with-example
	"(proceed (MSE (randn `(10 10)) (randn `(10 10))))"))
    
    (with-nn-doc 'cross-entropy-loss 'function)
    (with-nn-doc 'softmax-cross-entropy 'function)

    (with-nn-doc (find-class 'LinearLayer) 't
      (with-example
	"(LinearLayer 10 5)"))

    (with-nn-doc (find-class 'Conv2D) 't
      (with-example
	"(LinearLayer 3 5 '(3 3))"))))


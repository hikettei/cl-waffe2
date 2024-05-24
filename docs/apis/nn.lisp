
(in-package :cl-waffe2.docs)

(with-page *nn* "cl-waffe2/nn"
  (macrolet ((with-nn-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))
    (insert "## [Non Linear Activations]~%")
    
    (with-nn-doc '!relu 'function
      (with-example
	"(proceed (!relu (randn `(10 10))))"))

    (with-nn-doc '!gelu 'function
      (with-example
	"(proceed (!relu (randn `(10 10))))"))

    (with-nn-doc '!sigmoid 'function
      (with-example
	"(proceed (!sigmoid (randn `(10 10))))"))

    (with-nn-doc '!leaky-relu 'function
      (with-example
	"(proceed (!leaky-relu (randn `(10 10))))"))

    (with-nn-doc '!elu 'function
      (with-example
	"(proceed (!leakey-relu (randn `(10 10))))"))

    (with-nn-doc '!softmax 'function
      (with-example
	"(proceed (!softmax (randn `(3 3))))"))

    (insert "## [Normalization Layers]~%")

    (with-nn-doc (find-class 'BatchNorm2d) 't)
    (with-nn-doc (find-class 'LayerNorm2d) 't)

    (insert "## [Loss Functions]

### Tips: Utility Function

The `:reduction` keyword for all loss functions is set to T by default. If you want to compose several functions for reduction (e.g. ->scal and !sum), it is recommended to define utilities such as:

```lisp
(defun criterion (criterion X Y &key (reductions nil))
  (apply #'call->
	 (funcall criterion X Y)
	 (map 'list #'asnode reductions)))

;; Without criterion:
(->scal (MSE x y :reduction :sum))

;; With criterion for example:
(criterion #'MSE x y :reductions `(#'!sum #'->scal))
```
")
    
    (with-nn-doc 'L1Norm 'function
      (with-example
	"(proceed (L1Norm (randn `(10 10)) (randn `(10 10))))"))
    
    (with-nn-doc 'MSE 'function
      (with-example
	"(proceed (MSE (randn `(10 10)) (randn `(10 10))))"))
    
    (with-nn-doc 'cross-entropy-loss 'function)
    
    (with-nn-doc 'softmax-cross-entropy 'function)

    (insert "## [Linear Layers]~%")
    
    (with-nn-doc (find-class 'LinearLayer) 't
      (with-example
	"(LinearLayer 10 5)"))

    (insert "## [Dropout Layers]~%")
    
    
    (insert "## [Sparse Layers]~%")
    

    (insert "## [Recurrent Layers]~%")

    
    (insert "## [Convolutional Layers]~%")

    (with-nn-doc (find-class 'Conv2D) 't
      (with-example
	"(Conv2D 3 5 '(3 3))"))

    (insert "## Pooling Layers~%")

    (with-nn-doc (find-class 'MaxPool2D) 't)
    (with-nn-doc (find-class 'AvgPool2D) 't)
    (with-nn-doc 'unfold 'function)))


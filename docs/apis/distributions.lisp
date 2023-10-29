
(in-package :cl-waffe2.docs)

;; Here, place the docs about initializer function's family

;; TODO: Insert Some Examples.
(with-page *distributions* "Distributions"
  (with-section "Sampling matrices from distribution"
    (insert "cl-waffe2 provides a package `:cl-waffe2/distributions` which is used to sample matrices from the distributions.")

    (with-section "Common Format to the APIs"
      (insert "All sampling functions are defined in the following format via `define-tensor-initializer` macro.

```(function-name shape [Optional Arguments] &rest args &keys &allow-other-keys)```

That is, arguments passed to the `make-tensor` function can also be passed directly to the initializer functions.
")
      (with-example
	"(normal `(10 10) 0.0 1.0 :requires-grad t)")

      (with-example
	"(ax+b `(10 10) 1 0 :dtype :uint8)")

      (macrolet ((with-dist-doc (name type &body body)
		   `(with-section (format nil "~(~a~)"(symbol-name ,name))
		      (placedoc ,name ,type)
		      ,@body)))

	(with-dist-doc 'define-tensor-initializer 'macro)
	
	(with-dist-doc 'ax+b       'function
	  (with-example
	    "(ax+b `(3 3) 1.0 0.0)"))
	(with-dist-doc 'beta       'function
	  (with-example
	    "(beta `(3 3) 5.0 1.0)"))
	(with-dist-doc 'bernoulli  'function
	  (with-example
	    "(bernoulli `(3 3) 0.3)"))
	(with-dist-doc 'chisquare   'function
	  (with-example
	    "(chisquare `(3 3) 1.0)"))
	(with-dist-doc 'exponential 'function
	  (with-example
	    "(exponential `(3 3))"))
	(with-dist-doc 'gamma       'function
	  (with-example
	    "(gamma `(3 3) 1.0)"))
	(with-dist-doc 'normal      'function
	  (with-example
	    "(normal `(3 3) 1.0 0.0)"))
	(with-dist-doc 'uniform-random 'function
	  (with-example
	    "(uniform-random `(3 3) 2 4)"))
	(with-dist-doc 'randn 'function
	  (with-example
	    "(randn `(3 3))"))
	;; TO Add: orthogonal

	))))

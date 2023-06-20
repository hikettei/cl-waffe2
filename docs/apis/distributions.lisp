
(in-package :cl-waffe2.docs)

;; Here, place the docs about initializer function's family

;; TODO: Insert Some Examples.
;; サンプルコードを出力しながら書いていく
;; こっちにドキュメントの本体を書く

;; NewLine
(with-page *distributions* "Distributions"
  (with-top "Samples matrices from distribution"
    (insert "In order to create new matrices from distribution, cl-waffe2 provides a package, `:cl-waffe2/distributions`.")

    (with-top "Common Format to the APIs"
      (insert "All sampling functions are defined in the following format via `define-tensor-initializer` macro.

```(function-name shape [Optional Arguments] &rest args &keys &allow-other-keys)```

`Optional Arguments` will be passed to the function `make-tensor`, accordingly, both of these functions are valid for example.

1. (normal `(10 10) 0.0 1.0 :dtype :double)
2. (normal `(10 10) 0.0 1.0 :requires-grad t)")

      (macrolet ((with-dist-doc (name type &body body)
		   `(with-top (symbol-name ,name)
		      (placedoc ,name ,type)
		      ,@body)))

	(with-dist-doc 'define-tensor-initializer 'macro)
	
	(with-dist-doc 'ax+b       'function)
	(with-dist-doc 'beta       'function)
	(with-dist-doc 'bernoulli  'function)
	(with-dist-doc 'chisquare   'function)
	(with-dist-doc 'expotential 'function)
	(with-dist-doc 'gamma       'function)
	(with-dist-doc 'normal      'function)
	(with-dist-doc 'uniform-random 'function)
	(with-dist-doc 'randn 'function)
	;; TO Add: orthogonal

	))))

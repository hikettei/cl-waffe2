
(in-package :cl-waffe2.docs)

;; Here, place the docs about initializer function's family

;; TODO: Insert Some Examples.
(with-page *distributions* "Distributions"
  (with-section "Samples matrices from distribution"
    (insert "In order to create new matrices from distribution, cl-waffe2 provides a package, @c(cl-waffe2/distributions).")

    (with-section "Common Format to the APIs"
      (insert "All sampling functions are defined in the following format via @c(define-initializer-function) macro.")

      (insert "@c((function-name shape [Optional Arguments] &rest args &keys &allow-other-keys))")

      (insert "@c(args) is a arguments which passed to make-tensor, accordingly, both of these functions are valid for example.")
      (with-enum 
	  (item "(normal `(10 10) 0.0 1.0 :dtype :double)")
          (item "(normal `(10 10) 0.0 1.0 :requires-grad t)"))))


  (macrolet ((with-dist-doc (name type &body body)
	       `(with-section ,name
		  (placedoc
		   "cl-waffe2/distributions"
		   ,type
		   ,name)
		  ,@body)))

    (with-dist-doc "define-initializer-function" "macro")

    
    (with-dist-doc "ax+b" "function")
    (with-dist-doc "beta" "function")
    (with-dist-doc "bernoulli" "function")
    (with-dist-doc "chisquare" "function")
    (with-dist-doc "expotential" "function")
    (with-dist-doc "gamma" "function")
    (with-dist-doc "normal" "function")
    (with-dist-doc "uniform-random" "function")
    (with-dist-doc "randn" "function")
    ;; TO Add: orthogonal

    ))

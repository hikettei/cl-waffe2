
(in-package :cl-waffe2/base-impl)

(defnode (MatMulNode (myself)
	  :where `(A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :slots nil
	  :documentation ""))

(defun !matmul (x y)
  ""
  )


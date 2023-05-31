
(in-package :cl-waffe2/base-impl)

(defnode (2DMatmulNode (myself)
	  :where `([i j] [j k] [i k] -> [i k])
	  :slots nil
	  :documentation ""))

(defun !matmul (x y)
  ""
  )



(in-package :cl-waffe2/nn)

;; Provides A BASIC APIs for Regression
;;
;; LinearLayer DenseLayer L1/L2 Norm Etc...
;;


;; TODO: Broadcasting Axes...
;; (!view tensor :~ t t)
(defun step-linear (weight x &optional bias)
  "y = x[batch_size, dim2] @ weight[batch_size, dim1].T + bias"
  (!add (!matmul x (!t weight))
	(!view bias `(:broadcast ,(car (shape x))) t)))


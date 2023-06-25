
(in-package :cl-waffe2/nn)

;; Provides A BASIC APIs for Regression
;;
;; LinearLayer DenseLayer L1/L2 Norm Etc...
;;

(defmodel (LinearLayer (self in-features out-features &optional (use-bias? t))
	   :slots ((weights :accessor linear-weight)
		   (bias    :initform nil :accessor linear-bias))
	   :where ([~ batch-size in-features] -> [~ batch-size out-features])
	   :on-call-> call-linear
	   :documentation "In-Features -> Out-Features")

  ;; Initialize Weights
  (setf (linear-weight self)
	(parameter (!mul 0.01 (randn `(,out-features ,in-features)))))
  
  ;; Init with Xavier
  (when use-bias?
    (setf (linear-bias self)
	  (make-tensor `(,out-features)
		       :requires-grad t))))

;; TODO:
;; Broadcasting (with regard to axes)
;; Printing Model Objects like cl-waffe
;; TODO: Broadcasting Axes...
;; (!view tensor :~ t t)
(defun step-linear (x weight &optional bias)
  "y = x[batch_size, dim2] @ weight[batch_size, dim1].T + bias"
  (if bias
      (!add (!matmul x (!t (!flexible weight))) (!flexible bias))
      (!matmul x (!t (!flexible weight)))))

;; TODO: Print-Object


(defmethod call-linear ((self LinearLayer) x)
  (step-linear
   x
   (linear-weight self)
   (linear-bias self)))


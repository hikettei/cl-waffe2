
(in-package :cl-waffe2/nn)

;; Provides A BASIC APIs for Regression
;;
;; LinearLayer DenseLayer L1/L2 Norm Etc...
;;


;; TODO:
;; Broadcasting (with regard to axes)
;; Printing Model Objects like cl-waffe
;; TODO: Broadcasting Axes...
;; (!view tensor :~ t t)
(defun step-linear (weight x &optional bias)
  "y = x[batch_size, dim2] @ weight[batch_size, dim1].T + bias"
  (if bias
      (!add (!matmul x (!t weight))
	    (!view bias t `(:broadcast ,(second (shape weight)))))
      (!matmul x (!t weight))))

;; TODO: Print-Object
(defmodel (LinearLayer (self in-features out-features &optional (use-bias? t))
	   :slots ((weights :accessor linear-weight)
		   (bias    :initform nil :accessor linear-bias))
	   :on-call-> call-linear
	   :documentation "In-Features -> Out-Features")
  (setf (linear-weight self)
	(make-tensor `(,in-features ,out-features)
		     :requires-grad t))

  ;; Init with Xavier
  (when use-bias?
    (setf (linear-bias self)
	  (make-tensor `(1 ,out-features)
		       :requires-grad t))))

(defmethod call-linear ((self LinearLayer) x)
  (step-linear
   (linear-weight self)
   x
   (linear-bias self)))



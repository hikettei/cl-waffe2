
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
(defun step-linear (x weight &optional bias)
  "y = x[batch_size, dim2] @ weight[batch_size, dim1].T + bias"
  (if bias
      (!add (!matmul x (!flexible weight)) (!flexible bias))
      (!matmul x (!flexible weight))))

;; TODO: Print-Object
(defmodel (LinearLayer (self in-features out-features &optional (use-bias? nil))
	   :slots ((weights :accessor linear-weight)
		   (bias    :initform nil :accessor linear-bias))
	   :on-call-> call-linear
	   :documentation "In-Features -> Out-Features")
  (setf (linear-weight self)
	(randn `(,in-features ,out-features)
	       :requires-grad t))

  ;; Init with Xavier
  (when use-bias?
    (setf (linear-bias self)
	  (make-tensor `(1 ,out-features)
		       :requires-grad t))))

(defmethod call-linear ((self LinearLayer) x)
  (step-linear
   x
   (linear-weight self)
   (linear-bias self)))

#|
(defun test ()
  (let ((x (randn `(100 784)))
	(layer1 (LinearLayer 784 512))
	(layer2 (LinearLayer 512 128))
	(layer3 (LinearLayer 128  10)))
    (cl-waffe2:with-config (:device :cpu
			    :no-grad nil)
      (let ((f (!sum (call layer3 (!sin (call layer2 (!sin (call layer1 x))))))))
	(multiple-value-bind (forward backward vars params) (build f)

	  ;;(sb-profile:profile "CL-WAFFE2/BACKENDS.CPU"
	  ;;		      "CL-WAFFE2/BACKENDS.LISP")
	  
	  (time (print (funcall forward)))
	  ;; Add? side effects. no effs on params
	  (time (print (funcall forward)))
	  ;;(sb-profile:report)
	  ;;(sb-profile:unprofile "CL-WAFFE2/BACKENDS.CPU"
	;;		      "CL-WAFFE2/BACKENDS.LISP")

	  )))))

|#

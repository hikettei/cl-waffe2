
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
      (!add (!matmul x (!t (!flexible weight))) (!flexible bias))
      (!matmul x (!t (!flexible weight)))))

;; TODO: Print-Object
(defmodel (LinearLayer (self in-features out-features &optional (use-bias? t))
	   :slots ((weights :accessor linear-weight)
		   (bias    :initform nil :accessor linear-bias))
	   :where ([~ batch-size in-features] -> [~ batch-size out-features])
	   :on-call-> call-linear
	   :documentation "In-Features -> Out-Features")
  (setf (linear-weight self)
	(parameter (!mul 0.01 (randn `(,out-features ,in-features)))))
  
  ;; Init with Xavier
  (when use-bias?
    (setf (linear-bias self)
	  (make-tensor `(,out-features)
		       :requires-grad t))))

(defmethod on-print-object ((model LinearLayer) stream)
  (when (not (composite-traced-p model))
    (format stream "
    <Input: ((~~ BATCH-SIZE ~a)) -> Output: ((~~ BATCH-SIZE ~a))>
"
	    (car (shape (linear-weight model)))
	    (second (shape (linear-weight model))))))


(defmethod call-linear ((self LinearLayer) x)
  (step-linear
   x
   (linear-weight self)
   (linear-bias self)))

;; softmax with broadcasting

;; TMP
;; exp(x1) + exp(x2) + ... / exp(x1 + x2 + ...)
(defun !softmax (x)
  (let* ((x1 (!sub x (!mean x  :axis 1 :keep-repeat t)))
	 (z  (!sum   (!exp x1) :axis 1 :keep-repeat t)))
    (!div (!exp x1) z)))

;; with broadcasting...
(defun !softmax1 (x)
  (let* ((x (A-=B x (!mean x :axis 1 :keep-repeat t)))
	 (z-out (make-tensor `(,@(butlast (shape x)) 1)))
	 (subscripts `(,@(loop for i in (butlast (shape x))
			       collect t)
		       (:broadcast ,(car (last (shape x))))))
	 ;; To Add: (!view zout :~ `(:broadcast 10))
	 (z (!exp x :-> (apply #'!view z-out subscripts))))
    (!div (!exp x) z)))
#|
(defun test ()
  (let ((x (randn `(100 784) :requires-grad t))
	(layer1 (LinearLayer 784 512))
	(layer2 (LinearLayer 512 128))
	(layer3 (LinearLayer 128  10)))
    (cl-waffe2:with-config (:device :cpu
			    :no-grad nil)
      (let ((f (!softmax (call layer3 (!sin (call layer2 (!sin (call layer1 x))))))))
	(multiple-value-bind (forward backward vars params) (build f)

	  ;;(sb-profile:profile "CL-WAFFE2/BACKENDS.CPU"
	  ;;		      "CL-WAFFE2/BACKENDS.LISP")
	  
	  (time (print (funcall forward)))
	  ;; Add? side effects. no effs on params
	  (time (funcall backward))
	  (print (grad x))
	  (time (print (funcall forward)))
	  ;;(sb-profile:report)
	  ;;(sb-profile:unprofile "CL-WAFFE2/BACKENDS.CPU"
	;;		      "CL-WAFFE2/BACKENDS.LISP")

)))))
|#
  

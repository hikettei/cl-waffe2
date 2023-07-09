
(in-package :cl-waffe2/optimizers)

(defoptimizer (SGD (self param &key (lr 1e-3))
		   :slots ((lr :initarg :lr :reader sgd-lr))))

(defmodel (SGD-Compute-Form (self)
	   :where (Param[~] Lr[scal] -> Param[~] where scal = 1)
	   :documentation "Param_New <- Param - Param * Grad * Lr"
	   :on-call-> ((self param lr)
		       (declare (ignore self))
		       ;; Composite Function: Side Effects on Param?
		       (A-=B param (!mul lr param)))))

(define-composite-function (SGD-Compute-Form) step-sgd)

(defmethod step-optimize ((optimizer SGD))
  (let ((lr    (make-tensor (sgd-lr optimizer)))
	(param (grad (read-parameter optimizer))))
    
    (step-sgd param lr)))


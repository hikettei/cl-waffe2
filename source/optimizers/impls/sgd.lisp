
(in-package :cl-waffe2/optimizers)

(defoptimizer (SGD (self param &key (lr 1e-3))
		   :documentation "
### Inputs

Implements a simple SGD.

```math
Param_{new}\\gets{Param - Param_{grad}\\times{lr}}
```

`lr[single-float]` learning rate."
		   :slots ((lr :initarg :lr :reader sgd-lr))))

(defmodel (SGD-Compute-Form (self)
	   :where (Param[~] Grad[~] Lr[scal] -> Param[~] where scal = 1)
	   :documentation "Param_New <- Param - Param * Grad * Lr"
	   :on-call-> ((self param grad lr)
		       (declare (ignore self))
		       ;; Composite Function: Side Effects on Param?
		       (A-=B param (!mul lr grad)))))

(define-composite-function (SGD-Compute-Form) step-sgd)

(defmethod step-optimize ((optimizer SGD))
  (let* ((lr    (make-tensor (sgd-lr optimizer)))
	 (param (read-parameter optimizer))
	 (grad  (grad param)))
    (step-sgd param grad lr)))


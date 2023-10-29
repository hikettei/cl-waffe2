
(in-package :cl-waffe2/nn)

;; Provides A BASIC API for Regression
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

(defmodel (LinearLayer (self in-features out-features &optional (use-bias? t))
	   :slots ((weights :accessor linear-weight)
		   (bias    :initform nil :accessor linear-bias))
	   :where ([~ batch-size in-features] -> [~ batch-size out-features])
	   :on-call-> ((self x)
		       (step-linear
			x
			(linear-weight self)
			(linear-bias self)))
			
	   :documentation "Applies a linear transformation to the incoming data.

```math
y = xA^\\intercal + b
```

### Inputs

`in-features[fixnum]` size of each input size.

`out-features[fixnum]` size of each output size.

`bias[boolean]` If set to nil, the layer won't learn an additive bias. default: `t`

### Parameters

`(linear-weight self)` the trainable value of the model of shape `out_features * in_features`. The values are sampled from  `xavier-uniform`.

`(linear-bias self)` the trainable value of bias of shape `out_features`. If bias is t, the initial values are sampled from a uniform distribution: `U(-k, k)` where k = `sqrt(1/out-features)`.
")

  (setf (linear-weight self) (xavier-uniform `(,out-features ,in-features) :requires-grad t))
  
  (when use-bias?
    (let ((k (sqrt (/ 1 out-features))))
      (setf (linear-bias self) (uniform-random `(,out-features) (- k) k :requires-grad t)))))

;; [TODO] Rename: linear-weight -> linear-bias
;; They're alias, deleted in the future release.
(defmethod weight-of ((model LinearLayer))
  (linear-weight model))

(defmethod bias-of ((model LinearLayer))
  (linear-bias model))


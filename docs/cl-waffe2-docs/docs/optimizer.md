
# cl-waffe2/optimizers

## [class] AbstractOptimizer

`AbstractOptimizer` is an Abstract class of all optimizing functions in cl-waffe2.

The optimizing operation is performed by calling a `(step-optimize optimizer)` method. The parameter to be optimized can be accessed by a `read-parameter` method. The new optimizer function can be defined via `defoptimizer` macro.

See also: `defoptimizer` `read-parameter` `step-optimize`
## [macro] defoptimizer

The macro `defoptimizer` defines a user-defined optimizer class which is a subclass of `AbstractOptimizer`

The class is dispatched one per parameter to be optimized and the method `step-optimize` is called each time an optimizing is performed.

### Input

`param` the tensor to be optimized is given as this argument. the tensor is stored in the `param` slot automatically, being accessed by a `read-parameter` method.

### Example

```lisp
(defoptimizer (SGD (self param &key (lr 1e-3))
		   :slots ((lr :initarg :lr :reader sgd-lr))))

(defmodel (SGD-Compute-Form (self)
	   :where (Param[~] Grad[~] Lr[scal] -> Param[~] where scal = 1)
	   :documentation "Param_New <- Param - Param * Grad * Lr"
	   :on-call-> ((self param grad lr)
		       (declare (ignore self))
		       (A-=B param (!mul lr grad)))))

(define-composite-function (SGD-Compute-Form) step-sgd)

(defmethod step-optimize ((optimizer SGD))
  (let* ((lr    (make-tensor (sgd-lr optimizer)))
	 (param (read-parameter optimizer))
	 (grad  (grad param)))    
    (step-sgd param grad lr)))
```

## [optimizer] SGD

### Initializer

```
(SGD param &KEY (LR 0.001))
```

### Description


### Inputs

Implements a simple SGD.

```math
Param_{new}\gets{Param - Param_{grad}\times{lr}}
```

`lr[single-float]` learning rate.
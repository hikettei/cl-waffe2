
# cl-waffe2/optimizers

## [class] AbstractOptimizer

AbstractTensors with `:requires-grad=t` can find their gradients with the `(backward (build toplevel))` function. AbstractOptimizer is a class which minimizes the value of `toplevel` subject to `(grad tensor)`. In cl-waffe2, we initialize one AbstractOptimizer for one AbstractTensor. Specifically, one is able to create a new AbstractOptimizer with the function `(name tensor &rest constructor-args)`, for example, `(adam (parameter (randn `(3 3))) :lr 1e-3)` to create a new Adam Optimizer, and can be tied to the tensor like: `(hook-optimizer! tensor abstract-optimizer)`. Users can define any optimizer algorithms with the `defoptimizer` macro. Optimizing tied tensors is performed by calling a `(step-optimize optimizer)` method. The parameter to be optimized can be accessed by a `read-parameter` method.

### Example: Hooks and calls the optimizer tied to the tensor.

```lisp
(let ((a (parameter (randn `(3 3)))))
    (hook-optimizer! a (Adam a))
    (call-optimizer! a))
```

See also: `defoptimizer` `read-parameter` `step-optimize`.

## [macro] defoptimizer

The macro `defoptimizer` defines a user-defined optimizer class which is a subclass of `AbstractOptimizer`. And the class is dispatched one per parameter to be optimized and the method `step-optimize` is called each time an optimizing is performed.

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

(defmodel-as (SGD-Compute-Form) :named step-sgd)

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
## [optimizer] ADAM

### Initializer

```
(ADAM param &KEY (LR 0.001) (EPS 1.0e-7) (BETA1 0.9) (BETA2 0.999))
```

### Description


### Inputs

Implements Adam algorithm.

See the [original paper](https://arxiv.org/abs/1412.6980) for detailed algorithms.

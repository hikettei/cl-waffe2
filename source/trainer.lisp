
(in-package :cl-waffe2)

(defmacro hooker (bind optimizer &aux (opt-bind (gensym)))
  "
## [macro] hooker

```lisp
(hooker bind optimizer)
```

A convenient macro to hook AbstractOptimizers to each AbstractTensor. As the most straightforward explanation: this macro is expanded into this form.

```lisp
`(lambda (,bind)
     (hook-optimizer! ,bind ,optimizer))
```

where `bind` is excepted to be AbstractTensor, optimizer is a creation form of `AbstractOptimizer`, and the function `hook-optimizer!` hooks the given optimizer into bind.

In cl-waffe2, one independent Optimizer must be initialised per parameter. This macro can be used to concisely describe the process of initialising the same Optimiser for many parameters.

### Example

```lisp
;; (model-parameters compiled-composite) to read the list of all parameters in the network

(let ((model (build (!matmul 
		     (parameter (randn `(3 3)))
		     (parameter (randn `(3 3)))))))

  (mapc (hooker x (Adam X :lr 1e-3)) (model-parameters model))
  
  (forward model)
  (backward model)

  (mapc #'call-optimizer! (model-parameters model)))
```
"

  `(lambda (,bind)
     (declare (AbstractTensor ,bind))
     (let ((,opt-bind ,optimizer))
       (assert (typep ,opt-bind 'cl-waffe2/optimizers:AbstractOptimizer)
	   nil
	   "hooker: Assertion failed because the given optimizer isn't a subclass of AbstractOptimizer.
~a"
	   ,opt-bind)
       (hook-optimizer! ,bind ,opt-bind))))


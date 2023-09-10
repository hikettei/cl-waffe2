
(in-package :cl-waffe2/optimizers)

(defclass AbstractOptimizer ()
  ((param :initarg :param :type AbstractTensor :reader read-parameter))
  (:documentation "
## [class] AbstractOptimizer

AbstractTensors with `:requires-grad=t` can find their gradients with the `(backward (build toplevel))` function. AbstractOptimizer is a class which minimizes the value of `toplevel` subject to `(grad tensor)`. In cl-waffe2, we initialize one AbstractOptimizer for one AbstractTensor. Specifically, one is able to create a new AbstractOptimizer with the function `(name tensor &rest constructor-args)`, for example, `(adam (parameter (randn `(3 3))) :lr 1e-3)` to create a new Adam Optimizer, and can be tied to the tensor like: `(hook-optimizer! tensor abstract-optimizer)`. Users can define any optimizer algorithms with the `defoptimizer` macro. Optimizing tied tensors is performed by calling a `(step-optimize optimizer)` method. The parameter to be optimized can be accessed by a `read-parameter` method.

### Example: Hooks and calls the optimizer tied to the tensor.

```lisp
(let ((a (parameter (randn `(3 3)))))
    (hook-optimizer! a (Adam a))
    (call-optimizer! a))
```

See also: `defoptimizer` `read-parameter` `step-optimize`.
"))

(defgeneric step-optimize (optimizer))

;; One Parameter, -> One Optimizer class
(defmacro defoptimizer ((name
			 (self param &rest constructor-args)
			 &key
			   (documentation "")
			   (slots))
			&body constructor-body)
  "
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
	   :documentation \"Param_New <- Param - Param * Grad * Lr\"
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
"
  ;; :initarg :A 
  (let ((initargs (collect-initarg-slots slots `(,param ,@constructor-args))))
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (defclass ,name (AbstractOptimizer)
	 ,slots
	 (:documentation ,(format nil "
## [optimizer] ~a

### Initializer

```
(~a param~a)
```

### Description

~a"
				  name
				  name
				  (with-output-to-string (out)
				    (dolist (arg constructor-args)
				      (princ " " out)
				      (format out "~a" arg)))
				  documentation)))
       
       (defun ,name (,param ,@constructor-args)
	 ""
	 (assert (slot-value ,param 'requires-grad)
		 nil
		 "(~a) Failed with: the given parameter ~a isn't parameter."
		 ',name
		 ,param)
	 
	 (let ((,self (make-instance ',name
				     :param ,param
				     ;; Expand :slots
				     ,@(loop for slot in initargs
					     if slot
					       collect (intern (symbol-name (nth (1+ (position :initarg slot)) slot)) "KEYWORD")
					     if slot
					       collect (car slot)))))
	   ,@constructor-body
	   ,self)))))

(defmethod print-object ((opt AbstractOptimizer) stream)
  (format stream "<AbstractOptimizer: ~a(
    minimize   : toplevel
    subject to : <~a>~a
)>"
	  (class-name (class-of opt))
	  (tensor-id (read-parameter opt))
	  (cl-waffe2/vm.nodes::describe-tensor (read-parameter opt))))


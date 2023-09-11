
(in-package :cl-waffe2)

;; AbstractTrainer is a one of things to be deleted in the future release due to its complicaed notation and ugly format.

(defclass AbstractTrainer ()
    nil
  (:documentation "
## [class] AbstractTrainer

AbstractTrainer is an abstract definition of `Trainer` in cl-waffe2.

`Trainer` in cl-waffe2 bundles these features in one data structure:

1. Describe Trainining Step (including measuring criterion parts, predicting parts)

2. Describe Training Parameters in one structure.

3. Manages training mode/predicting mode

For general use, trainer can be defined by the macro `deftrainer`."))

(defmacro with-model-parameters ((bind model) &body body)
  `(let ((,bind (nodevariables-parameters
		 (compiled-variables ,model))))
     ,@body))

(defun initialize-optimizers! (model initializer-function)
  ""
  (declare (type Compiled-Composite model)
	   (type function initializer-function))
  (with-model-parameters (params model)
    (mapc #'(lambda (p)
	      (hook-optimizer! p (funcall initializer-function p)))
	  params)))

(defun zero-grads! (model)
  (declare (type Compiled-Composite model))
  (with-model-parameters (params model)
    (mapc #'(lambda (p)
	      (reset-grad! p))
	  params)))

(defun optimize! (model)
  (declare (type Compiled-Composite model))
  (with-model-parameters (params model)
    (mapc #'call-optimizer! params)))

(defmacro deftrainer ((name (self &rest constructor-arguments)
		       &key
			 (model)
			 (optimizer '(SGD :lr 1e-3))
			 (slots)
			 (documentation "")
			 (compile-mode :fastest)
			 (build)
			 (minimize!)
			 (set-inputs)
			 (predict))
		      &body constructor-body)
  "
## [macro] deftrainer

defines a new trainer.

(TODO Docstring)
"

  (let ((initargs (collect-initarg-slots slots constructor-arguments))
	(model-gensym (gensym "model"))
	(last-node    (gensym))
	(inputs       (gensym "inputs")))
    `(progn
       (defclass ,name (AbstractTrainer)
	 (,@slots
	  (compiled-model :initarg :compiled-model :reader compiled-model)
	  (model :initarg :model :reader model))
	 (:documentation ,documentation))

       (defmethod minimize! ((self ,name) &rest ,inputs)	 
	 (multiple-value-bind (,@(car minimize!)) (apply #'values `(,self ,@,inputs))
	   (declare (ignorable ,(caar minimize!)))
	   ,@(cdr minimize!)))

       (defmethod set-inputs ((self ,name) &rest ,inputs)
	 (multiple-value-bind (,@(car set-inputs)) (apply #'values `(,self ,@,inputs))
	   (declare (ignorable ,(caar set-inputs)))
	   ,@(cdr set-inputs)))

       (defmethod predict ((self ,name) &rest ,inputs)
	 (multiple-value-bind (,@(car predict)) (apply #'values `(,self ,@,inputs))
	   (declare (ignorable ,(caar predict)))
	   ,@(cdr predict)))

       (defun ,name (,@constructor-arguments)
	 (flet ((create-optimizer (parameter)
		  "Initializes an AbstractOptimizer for given parameter."
		  ,(let ((initform `(,(car optimizer) parameter ,@(cdr optimizer))))
		     initform))
		(,model-gensym (,@(car build))
		  ,@(cdr build)))
	   (let* ((,self (make-instance ',name
					:model ,model
					,@(loop for slot in initargs
						if slot
						  collect (intern (symbol-name (nth (1+ (position :initarg slot)) slot)) "KEYWORD")
						if slot
						  collect (car slot))))
		  (,last-node (,model-gensym ,self))
		  (,model-gensym (build ,last-node
					:compile-mode ,compile-mode))) ;; Build and compile models.
	     
	     (setf (slot-value ,self 'compiled-model) ,model-gensym) ;; Set Compiled-Model.
	     (initialize-optimizers! (compiled-model ,self) #'create-optimizer)
	     ,@constructor-body
	     ,self))))))

;; ↑ いつか消す
;; ~~~~~~~~~~~~~~~~~~~~~~~~


;; 今日やること
;; cl-waffe2/vmパッケージを綺麗にする + ドキュメント整備 + Trainer廃止
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



(in-package :cl-waffe2/optimizers)

(defclass AbstractOptimizer ()
  ((param :initarg :param :type AbstractTensor :reader read-parameter))
  (:documentation "
## [class] AbstractOptimizer

`AbstractOptimizer` is a CLOS class which

(TODO DOCS)

See also: `read-parameter` `step-optimize`"))

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

(TODO DOCS)
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
(~a param &rest arguments)
```

### Description

~a"
				  name
				  name
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


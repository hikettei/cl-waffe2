
(in-package :cl-waffe2)


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

defines a new trainer."

  (let ((initargs (collect-initarg-slots slots constructor-arguments))
	(model-gensym (gensym "model"))
	(inputs      (gensym "inputs")))
    `(progn
       (defclass ,name (AbstractTrainer)
	 (,@slots
	  (model :initarg :model :reader model))
	 (:documentation ,documentation))

       (defmethod minimize! ((self ,name) &rest ,inputs)	 
	 (multiple-value-bind (,@(car minimize!)) (apply #'values `(,self ,@,inputs))
	   (declare (ignorable ,(caar minimize!)))
	   ,@(cdr minimize!)))

       (defmethod set-inputs ((self ,name) &rest ,inputs)
	 (multiple-value-bind (,@(car set-inputs)) (apply #'values `(,self ,@,inputs))
	   (declare (ignorable ,(caar minimize!)))
	   ,@(cdr set-inputs)))

       (defmethod predict ((self ,name) &rest ,inputs)
	 (multiple-value-bind (,@(car predict)) (apply #'values `(,self ,@,inputs))
	   (declare (ignorable ,(caar minimize!)))
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
		  (,model-gensym (build (,model-gensym ,self)
					:compile-mode ,compile-mode))) ;; Build and compile models.
	     
	     (setf (slot-value ,self 'model) ,model-gensym) ;; Set Compiled-Model.
	     (initialize-optimizers! (model ,self) #'create-optimizer)
	     ,@constructor-body
	     ,self))))))

;; TODO Printer

;;(defmethod print-object ((model AbstractTrainer) stream)

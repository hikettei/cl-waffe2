
(in-package :cl-waffe2/vm.nodes)

(defclass Composite ()
  nil
  (:documentation "nn.Module"))

(defgeneric call (model &rest inputs) (:documentation ""))

(defmethod call :before ((model Composite) &rest inputs)
  (declare (ignore inputs))
  ;; Update States?
  (assert (subtypep (class-of model) 'Composite)
	  nil
	  "Assertion Failed with call method, because the model ~a isn't subtype of cl-waffe2/vm.nodes:Composite." model))

(defmacro define-forward-function (model forward-function)
  "forward-function = funcallable function"
  `(defmethod call ((model ,model) &rest inputs)
     (apply ,forward-function model inputs)))

;; forward-functionもGenericにする
;; Enhancement with multiple-dispatching
;; Generic: Conv2D Conv3D Conv4Dを自動で割り当てられる・・・

;; Callで本物の関数をWrapする

;; There's two way to build model
;; Extend AbstractModel
;; using defmethod

;; (defmacro defmodel

(defmacro defmodel ((name
		     (self-name &rest constructor-arguments)
		     &key
		       (slots nil)
		       (initargs)
		       (on-call-> nil)
		       (documentation ""))
		    &body constructor-body)
  "TODO: Docstring"
  (declare (type (or symbol function list null) on-call->))
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (prog1
	 ;; E.g.: The case when we want to define LinearLayer Model...
	 ;; defines LinearLayer class
	 (defclass ,name (Composite)
	   (,@slots)
	   (:documentation ,(format nil "class, ~a" documentation)))

       ;; Creates a constructor named (linearlayer constructor-arguments)
       (defun ,name (,@constructor-arguments)
	 ,(format nil "An constructor for model.")
	 (let ((,self-name (make-instance
			    ',name
			    ,@initargs)))
	   ,@constructor-body
	   ,self-name))

       ;; Registers forward funcition cl-waffe2 uses.
       ;; on-call-> could be one of them:
       ;; 1. method-name
       ;; 2. function-name
       ;; 3. compiled-function
       ;; 4. list (compiled automatically)
       ,(typecase on-call->
	  (null nil)
	  (symbol   `(define-forward-function ,name #',on-call->))
	  (function `(define-forward-function ,name ,on-call->))
	  (list     `(define-forward-function ,name #'(lambda ,@on-call->)))))))

#|
(defmodel (LinearLayer (self in-features out-features &key (bias t))
	   :slots ((weight :initarg :weight)
		   (bias   :initarg :bias))
	   :initargs (:weight (make-tensor  `(,in-features ,out-features))
		      :bias   (if bias
				  (make-tensor `(1 ,out-features))))))
|#


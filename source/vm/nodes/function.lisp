
(in-package :cl-waffe2/vm.nodes)

;; The file function.lisp provides a system on interacting Lazy-Evaluated Nodes and Toplevel functions:
;;
;; Composite <-> Function
;; Node      <-> Function
;;

;; 1. Composite Functionを作る
;; Issuesを解消する


(defmodel (DotProduct (self)
	   :where ([~] [~] -> [~])
	   :on-call-> ((self x y)
		       (declare (ignore self))
		       (cl-waffe2/base-impl:!sum (cl-waffe2/base-impl:!mul x y)))))

(defmodel (DotProduct-2D (self)
	   :where ([a b] [a b] -> [scal] where scal = 1)
	   :on-call-> ((self x y)
		       (declare (ignore self))
		       (cl-waffe2/base-impl:!sum (cl-waffe2/base-impl:!mul x y)))))

(defun composite->defun (~ composite function-name
			 &key
			   (dtype :float)
			   (order :column)
			   (scalar-p nil)
			   (compile-mode :default))
  "
## [function] composite->function

Tracing definition of given composite, the function `composite->function` returns a compiled-lambda function.

Return: (values input-names lambda-function)
"

  ;; Reading where
  ;; where as shape
  ;; set ~ = given ~ (NIL IS OK)

  ;; Tracing
  ;; Building
  ;; Lambda Encapsulate

  ;; Receives Input Tensors
  (let* ((inputs (composite-input-tensor composite ~
					 :dtype dtype
					 :order order
					 :scalar-p scalar-p))
	 (namelist (or (composite-symbol-names composite)
		       (loop for i upfrom 0 below (length inputs)
			     collect (nth-subscript i))))
	 (result (apply #'call composite inputs))
	 (self   (gensym))
	 (model-name (symb 'compiled-
			   function-name
			   '-
			   (intern (format nil "~a" dtype))
			   '-
			   (intern (format nil "~a" ~))))
	 (tmp-fname (gensym (symbol-name function-name)))
	 (mname     (gensym)))
    (with-no-grad
      (let ((compiled-kernel (cl-waffe2/vm.generic-tensor:build result :compile-mode compile-mode)))
	
	`(progn
	   ;; Model Caller
	   (defun ,function-name  (,@namelist)
	     (let ((,mname (,model-name)))
	       (shape-compatible? ,mname ,@namelist)
	       (call ,mname ,@namelist)))

	   ;; Main Body
	   (defun ,tmp-fname (,self ,@namelist)
	     (declare (ignore ,self))
	     ,@(loop for tensor in inputs
		     for name in namelist
		     collect `(set-input ,compiled-kernel ,(tensor-name tensor) ,name))
	     (forward ,compiled-kernel))

	   (defmodel (,model-name (self)
		      :where ,(read-where composite)
		      :on-call-> ,tmp-fname)))))))

(defun composite->generic (composite function-name)

  )


(defmacro define-composite-solid-function (composite-init-form function-name)
  "
## [macro] define-composite-solid-function

"

  `(eval (composite->defun nil ,composite-init-form ',function-name)))

(define-composite-solid-function (DotProduct-2D) !dot2d)

(define-composite-det-function (DotProduct) !dotproduct)

;; define-composite-undetermined-function
;; define-composite-function

  
(defun test-composite-f ()
  (let ((model (DotProduct)))
    (composite->function
     (dim->input-shape 2)
     model)))

(defmacro def-composite->function (composite-name function-name)
  "
## [macro] composite->function

The macro `define-composite-function` traces the computation node of given Composite, `composite-name`, defining a function."

  ;; Is there ~ parameter?
  ;; If True -> We need a method in order to dispatch the appropriate dim
  ;; Otherwise -> We need a single function

  
  
  
  
  )


(defmacro define-as-operation (toplevel)
  "
## [macro] define-as-operation
AsNode but defines the function TOPLEVEL.
"
  )

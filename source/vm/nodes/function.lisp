
(in-package :cl-waffe2/vm.nodes)

;; The file function.lisp provides a system on interacting Lazy-Evaluated Nodes and Toplevel functions:
;;
;; Composite <-> Function
;; Node      <-> Function
;;

;; TODO:
;; 1. Composite Functionを作る
;; Issuesを解消する


(defmodel (DotProduct (self)
	   :where ([~] [~] -> [~])
	   :on-call-> ((self x y)
		       (declare (ignore self))
		       (cl-waffe2/base-impl:!sum (cl-waffe2/base-impl:!mul x y)))))

(defmodel (DotProduct-2D (self)
	   :where (A[a b] B[a b] -> [scal] where scal = 1)
	   :on-call-> ((self x y)
		       (declare (ignore self))
		       (cl-waffe2/base-impl:!sum (cl-waffe2/base-impl:!mul x y)))))


;; One composite -> a single defun form.
;; In order to implemenet "GENERIC" behaviour, we need to wrap composite->defun by higher-order function.
(defun composite->defun (~ composite function-name
			 &key
			   (dtype :float)
			   (order :column)
			   (scalar-p-list nil)
			   (compile-mode :default))
  "
## [function] composite->function

Tracing definition of given composite, the function `composite->function` returns a defun forms.

[One ~ value, one :dtype, one scalar-p-list state -> single defun form.]

dtype/scalar-p-list = list is ok

Return: defun form
"


  (let* ((inputs (composite-input-tensor composite ~
					 :dtype dtype
					 :order order
					 :scalar-p-list scalar-p-list))
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
	   ;; Main Body
	   (defun ,tmp-fname (,self ,@namelist)
	     (declare (ignore ,self))
	     ,@(loop for tensor in inputs
		     for name in namelist
		     collect `(set-input ,compiled-kernel ,(tensor-name tensor) ,name))
	     (forward ,compiled-kernel))

	   (defmodel (,model-name (self)
		      :where ,(read-where composite)
		      :on-call-> ,tmp-fname))

	   (defun ,function-name  (,@namelist)
	     (let ((,mname (,model-name)))
	       (shape-compatible? ,mname ,@namelist)
	       (call ,mname ,@namelist)))
		   
	   #',function-name)))))

(defun ->key (&rest inputs)
  (apply #'symb (map 'list #'tensor-keyname inputs)))

;; dispatch with ~
;; set dtype
(defun dispatch-method (polymorphic-table
			inputs
			composite
			function-name
			~state
			&key 
			  (compile-mode :default)
			  (order :column))

  (let* ((key (apply #'symb
		     (map 'list #'->key inputs)))
	 (key (if ~state
		  (symb key (intern (format nil "~a" ~state)))
		  key))
	 (~state (loop for i upfrom 0
		       for ~s in ~state
		       collect (dim->input-shape ~s))))

    (or (gethash key polymorphic-table)
	(let* ((defun-form (composite->defun
			    ~state
			    composite
			    (symb function-name key)
			    :dtype (map 'list #'dtype inputs)
			    :order order
			    :scalar-p-list (map 'list #'scalar-p inputs)
			    :compile-mode compile-mode))
	       (compiled-f (eval defun-form)))
	  (setf (gethash key polymorphic-table) compiled-f)
	  compiled-f))))

(defun composite->generic (composite function-name
			   &key
			     (adjustable-size t)
			     (compile-mode :default)
			     (order :column))
  "Polymorphic dispatching kernel"
  (let ((polymorphic-table (make-hash-table))
	(kernel-place (gensym (format nil "~a-state-" function-name)))
	(namelist (or (composite-symbol-names composite)
		      (loop for i upfrom 0 below (length (composite-input-size composite))
			    collect (nth-subscript i))))
	(target-function (gensym))
	(input-det-n-list (input-det-n-list composite))
	(~length-place (gensym))) ;; the number of subscripts list...
    
    `(progn
       (defparameter ,kernel-place ,polymorphic-table)

       (defun ,function-name (,@namelist)
	 (let* ((,~length-place ,(when adjustable-size
				   `(map 'list #'(lambda (x y) (- (cl-waffe2/vm.generic-tensor:dims x) y)) (list ,@namelist) (list ,@input-det-n-list))))
		(,target-function (dispatch-method
				   ,kernel-place (list ,@namelist)
				   ,composite ',function-name
				   ,~length-place
				   :compile-mode ,compile-mode
				   :order ,order)))
	   (funcall ,target-function ,@namelist))))))

(defun composite-function (composite
			   function-name
			   &key
			     (compile-mode :default)
			     (scalar-p-list nil)
			     (order :column)
			     (dtype t))
  "
## [function] composite-function

Returns an defun form that can be evaluated by (eval ...).

On the condition where composite should be defined as polymorphic, returns generic definition/dispatching.

On the condition where composite should be defined as normal defun, returns a single defun form.

Input:
   dtype[t or keyword] If t, creates a hash-table that can be all definiton on dtype is stored.
                       If keyword, a single definition of the given dtype is created.

   scalar-p-list [list] the corresponding position of argument is interpreted as a scalartensor.
"

  (when (null (read-where composite))
    (error "Failed to define composite-function, because :where form of ~a isn't declared yet." composite))

  (let ((including~? (include~p composite)))
    (cond
      ((and (keywordp dtype)
	    (null including~?))
       ;; A facet of function is determined as one.
       (composite->defun nil composite function-name
			 :dtype dtype
			 :order order
			 :compile-mode compile-mode
			 :scalar-p-list scalar-p-list))
      (T
       (composite->generic composite function-name
			   :adjustable-size including~?
			   :order order
			   :compile-mode compile-mode)))))


(defmacro define-composite-function (composite-init-form
				     function-name
				     &key
				       (dtype t)
				       (order :column)
				       (compile-mode :default))
  "
## [macro] define-composite-function



"
  (declare (type (or t keyword) dtype)
	   (type keyword order)
	   (type compile-option-t compile-mode))

  `(eval (composite-function ,composite-init-form ',function-name
			     :compile-mode ,compile-mode
			     :dtype ,dtype
			     :order ,order)))

(define-composite-function (DotProduct-2D) !dot2d-float :dtype :float)

(define-composite-function (DotProduct-2D) !dot2d)

(define-composite-function (DotProduct) !dotproduct)


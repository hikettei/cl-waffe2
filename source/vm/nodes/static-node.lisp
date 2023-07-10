
(in-package :cl-waffe2/vm.nodes)


;; Export
(defparameter *under-composite-node-mode* nil "Set t under :forward in composite-node. While this parameter is t, set-save-for-backward is never used.")

;; Implementation of save-for-backward/read-save-for-backward
;; Both of them are called with save-for-backward function binded by with-composite-node-mode
;;


(defun apply-set-save-for-backward (self name tensor)

  )

(defun apply-read-save-for-backward (self name)

  )


(defmacro with-composite-node-mode (&body body)
  "Under the body execution, Composite-Node-Mode is enabled which is:

1. save-for-backward/read-save-for-backward function become available.
2. *under-composite-node-mode* becomes t."
  `(let ((*under-composite-node-mode* t))
     (flet ((set-save-for-backward (self name tensor)
	      (apply-set-save-for-backward self name tensor))
	    (read-save-for-backward (self name)
	      (apply-read-save-for-backward self name)))
       #'set-save-for-backward
       #'read-save-for-backward
       ,@body)))

;; Binding a set-save-for-backward function at toplevel
(defun set-save-for-backward (self name tensor)
  (error "Attepted to ..."))

;; Binding a read-save-for-backward function at toplevel
(defun read-save-for-backward (self name)
  (error "should be called with with-composite-node-mode"))

(defmacro define-static-node ((name
			       (self-name &rest constructor-args)
			       &key
				 (where nil)
				 (slots nil)
				 (out-scalar-p nil)
				 (forward nil)
				 (backward nil)
				 (documentation ""))
			      &body constructor-body)
  "
## [macro] define-static-node

Defines a differentiable composite, instantly defines as composite-function
↑EncapsulatedNodeと同じ要領で実装できる"

  ;; TODO: Checking the number of arguments
  ;; TODO: set-save-for-backward
  ;; TODO: gradtmp space
  
  (let ((forward-args  (car forward))
	(backward-args (car backward))
	(forward-body  (cdr forward))
	(backward-body (cdr backward))
	(backward-node-name (symb name '-backward))

	(forward-flet-name (symb name '-forward-body))
	(backward-flet-name (symb name '-backward-body)))
    
    `(progn
       ;; Called only when backward-mode
       (define-and-impl-node (,backward-node-name (self forward-self)
			      :device t
			      ;; Temporary Restores Nodes used in forward.
			      :slots ((forward-self :initarg :forward-self :reader read-forward-self))
			      ;; forward -> backward => backward -> forward
			      :where ,(where->backward where)
			      :forward ((,@backward-args)
					;; Swapping (self args...) -> (forward-self args...)
					(flet ((,backward-flet-name (,@backward-args)
						 (declare (ignorable ,(car backward-args)))
						 (with-composite-node-mode
						   ,@backward-body)))
					  (multiple-value-bind (,@backward-args)
					      (apply #'values (list (read-forward-self ,(car backward-args)) ,@(cdr backward-args)))
					    `(funcall ,#',backward-flet-name ,,@backward-args))))))

       ;; This is a main part of composite-node.
       (define-and-impl-node (,name (,self-name ,@constructor-args)
			      :device t
			      :where ,where
			      :slots ,slots
			      :out-scalar-p ,out-scalar-p
			      :documentation ,documentation
			      :forward ((,@forward-args)
					(flet ((,forward-flet-name (,@forward-args)
						 (declare (ignorable ,(car forward-args)))
						 (with-composite-node-mode
						   ,@forward-body)))
					  `(funcall ,#',forward-flet-name ,,@forward-args)))
			      :backward ((,@backward-args)					 
					 (forward (,backward-node-name ,(car backward-args)) ,@(cdr backward-args))))
	 ,@constructor-body))))


(define-static-node (TestModel (self)
		     :where (A[~] -> OUT[~])
		     :forward ((self x)
			       (print "HI :)")
			       (cl-waffe2/base-impl:proceed
				(cl-waffe2/base-impl:!sin x)))
		     :backward ((self dout)
				(values
				 (cl-waffe2/base-impl:proceed
				  (cl-waffe2/base-impl:!cos dout))))))





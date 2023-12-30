
(in-package :cl-waffe2/backends.metal)


;; Allocator, Dependencies, Configuration for MetalTensor
(defclass MetalTensor (cl-waffe2/backends.lisp:LispTensor)
  nil
  (:documentation
   "## [AbstractTensor] MetalTensor
Provides Metal-Accelerated Operations
"))

(defmethod wf/t:current-backend-state ((backend-name (eql 'MetalTensor)))
  #+metal(format nil "Available (~a)" (machine-version))
  #-metal(format nil "Not Available"))

;; ~~ Utils for manipulating array pointers ~~~~~~~~~~~~~~~~~~~~~~~~~~~
(defmacro with-tensor-ptr ((bind tensor) &body body)
  `(let ((,bind (tensor-vec ,tensor)))
     ;; Stored as a simple-array
     ,@body))     

(defmacro with-tensor-ptrs ((&rest input-forms) &body body)
  (labels ((expand (rest-forms)
	     (if rest-forms
		 `(with-tensor-ptr (,(caar rest-forms) ,(second (car rest-forms)))
		    ,(expand (cdr rest-forms)))
		 `(progn ,@body))))
    (expand input-forms)))


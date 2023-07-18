
(in-package :cl-user)

(defpackage :cl-waffe2/optimizers
  (:use :cl :cl-waffe2/base-impl :cl-waffe2/vm.generic-tensor :cl-waffe2/vm.nodes :alexandria)
  (:export
   #:AbstractOptimizer
   #:defoptimizer
   #:read-parameter
   #:step-optimize)

  (:export
   #:SGD
   #:Adam))

(in-package :cl-waffe2/optimizers)


(defun collect-initarg-slots (slots constructor-arguments)
  (map 'list #'(lambda (slots)
		 ;; Auto-Generated Constructor is Enabled Only When:
		 ;; slot has :initarg
		 ;; slot-name corresponds with any of constructor-arguments
		 (when
		     (and
		      (find (first slots) (flatten constructor-arguments))
		      (find :initarg slots))
		   slots))
       slots))


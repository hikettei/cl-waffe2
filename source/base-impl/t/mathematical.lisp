
(in-package :cl-waffe2/base-impl.test)

(in-suite :base-impl-test)

;; == Here, we provide testing framework. ========
;;
;; You can perform tests like:
;;
;; (mathematical-test-set LispTensor)
;;
;; ===============================================

;; TODO: Tests on all mathematical kernels.
(macrolet ((define-mathematical-kernel-tester (name wf-op lisp-op bw-lisp)
	     `(define-tester ,name :dense
		(let ((a (make-tensor `(30 30) :initial-element 1)))
		  (let ((c (proceed (,wf-op a)))
			(all-p t))
		    (loop while all-p
			  for i upfrom 0 below 900
			  do (setq all-p (= (vref c i) (funcall ,lisp-op 1))))
		    (when all-p
		      (let ((a (make-tensor `(30 30) :initial-element 1 :requires-grad t)))
			(proceed-backward (,wf-op a))
			(let ((all-p t))
			  (loop while all-p
				for i upfrom 0 below 900
				do (setq all-p (= (vref (grad a) i) (funcall ,bw-lisp 1))))
			  (if all-p
			      t
			      :backward)))))))))
  (define-mathematical-kernel-tester
      sin-tester
    !sin
    #'sin
    #'cos)
  )

(eval-when (:compile-toplevel :load-toplevel :execute)
  (export 'mathematical-test-set)
  (defmacro mathematical-test-set (backend)
    `(progn
       (sin-tester ,backend))))
			

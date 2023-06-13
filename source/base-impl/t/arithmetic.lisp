
(in-package :cl-waffe2/base-impl.test)

(in-suite :base-impl-test)

(macrolet ((define-arith-tester (name op result grad1 grad2)
	     `(define-tester ,name :all
		(let ((a (make-tensor `(100 100) :initial-element 10))
		      (b (make-tensor `(100 100) :initial-element 1)))
		  (let ((c (proceed (,op a b)))
			(all-p t))
		    (loop while all-p
			  for i fixnum upfrom 0 below 10000
			  do (setq all-p (= (vref c i) ,result)))
		    (when all-p
		      (let ((a (make-tensor `(100 100) :initial-element 10 :requires-grad t))
			    (b (make-tensor `(100 100) :initial-element 1 :requires-grad t)))
			(proceed-backward (,op a b))
			(let ((all-p t))
			  (loop while all-p
				for i fixnum upfrom 0 below 10000
				do (setq all-p (and (= (vref (grad a) i) ,grad1)
						    (= (vref (grad b) i) ,grad2))))
			  (if all-p
			    t
			    :backward)))))))))
  (define-arith-tester add-tester !add 11 1  1)
  (define-arith-tester sub-tester !sub 9  1 -1)
  (define-arith-tester mul-tester !mul 10 1 1)
  (define-arith-tester div-tester !div 10 1 -1))

;; TODO: Move to backends/cpu, backends/lisp
(add-tester cl-waffe2/backends.lisp:LispTensor)
(sub-tester cl-waffe2/backends.lisp:LispTensor)
(mul-tester cl-waffe2/backends.lisp:LispTensor)
(div-tester cl-waffe2/backends.lisp:LispTensor)

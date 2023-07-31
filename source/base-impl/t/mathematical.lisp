
(in-package :cl-waffe2/base-impl.test)

;; == Here, we provide testing framework. ========
;;
;; You can perform tests like:
;;
;; (mathematical-test-set LispTensor)
;;
;; ===============================================

;; TODO: Tests on all mathematical kernels.
;;
;; To check gradients, the most simple way would be on REPL, use proceed-backward:
;;
;; (let ((a (make-tensor `(10 10) :requires-grad t)))
;;     (proceed-backward (!sin a))
;;     (grad a))
;;

;; Testing:
;;
;; [X X X] <- <Mathematical Nodes> <- Input
;;    ↑ (Check) Is it correct?

;; [1 1 1] -> <Mathematical Nodes> -> Gradient
;;                                       ↑ (Check) Is it correct?

;;
;; TODO:
;; Tests of -> Scalar-And-Scalar Nodes
;; Tests of -> All math kernels
;; implement: expt node
;; Test     -> reshape proceed viewnode's backward unsqueeze copy broadcast(flexible)
;; Tests of -> ->scal, ->mat, 

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
      abs-tester
    !abs
    #'abs
    #'signum)


  (define-mathematical-kernel-tester
      sign-tester
    !sign
    #'signum
    #'(lambda (x) (* x 0)))

  (define-mathematical-kernel-tester
      sqrt-tester
    !sqrt
    #'signum
    #'(lambda (x) (/ 1 x)))

  (define-mathematical-kernel-tester
      square-tester
    !square
    #'(lambda (x) (* x x))
    #'(lambda (x) x))

  
  (define-mathematical-kernel-tester
      sin-tester
    !sin
    #'sin
    #'cos)

  (define-mathematical-kernel-tester
      cos-tester
    !cos
    #'cos
    #'(lambda (x) (- (sin x))))

  (define-mathematical-kernel-tester
      tan-tester
    !tan
    #'tan
    #'(lambda (x) (/ (expt (cos x) 2)))) ;; To Impl: ExptNode

  (define-mathematical-kernel-tester
      sinh-tester
    !sinh
    #'sinh
    #'cosh)

  (define-mathematical-kernel-tester
      cosh-tester
    !cosh
    #'cosh
    #'(lambda (x) (- (sinh x))))

  (define-mathematical-kernel-tester
      tanh-tester
    !tanh
    #'tanh
    #'(lambda (x) (/ (expt (cosh x) 2)))) ;; To Impl: ExptNode

  ;; To ADD: asin acos atan/ asinh acosh atanh

  (define-mathematical-kernel-tester
      exp-tester
    !exp
    #'exp
    #'exp)

  (define-mathematical-kernel-tester
      log2-tester
    !log2
    #'(lambda (x) (log x 2))
    #'(lambda (x) (/ 1 (* x (log 2)))))
  
  (define-mathematical-kernel-tester
      log10-tester
    !log10
    #'(lambda (x) (log x 10))
    #'(lambda (x) (/ 1 (* x (log 10)))))

  (define-mathematical-kernel-tester
      loge-tester
    !loge
    #'log
    #'(lambda (x) (/ 1 x)))
  )

(eval-when (:compile-toplevel :load-toplevel :execute)
  (export 'mathematical-test-set)
  (defmacro mathematical-test-set (&rest backend)
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (abs-tester ,@backend)
       (sign-tester ,@backend)
       (sqrt-tester ,@backend)
       (square-tester ,@backend)
       
       (sin-tester ,@backend)
       (cos-tester ,@backend)
       ;;(tan-tester ,backend)
       ;; To Add: trig func fam

       (exp-tester ,@backend)
       (log2-tester ,@backend)
       (log10-tester ,@backend)
       (logE-tester ,@backend)
       ;; add: expt
       )))
			
;; ScalarTest here.

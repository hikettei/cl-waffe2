
(in-package :cl-waffe2/backends.lisp)

;; LispKernel -> Safe First
;; (TO BE) Working with lparallel If needed.
;; TODO: (sin X:Int) -> OUT:Float
;; ==============================================================
;; F() -> G(X, OUT) Function Family
;; ==============================================================

;; Sparse/Dense Matrices should be separated together.
(macrolet ((define-math-impl (node-name one-arg-fn densep &aux (lambdap (listp one-arg-fn)))
	     "Sparse and Dense"
	     (let ((caller-name (symb node-name '-caller)))
	       `(eval-when (:compile-toplevel :load-toplevel :execute)
		  (,(if densep 'define-with-typevar-dense
			'define-with-typevar)
		   (,caller-name u)
		   (x y offsetx offsety incx incy size)
		   ;; Y <- F(X)
		   (declare (optimize (speed 3))
			    (type (simple-array u (*)) x y)
			    (type fixnum offsetx offsety incx incy size))
		   (dotimes (i size)
		     (setf (aref y (+ offsety (the fixnum (* incy i))))
			   ,(if lambdap
				`(funcall ,one-arg-fn (aref x (+ offsetx (the fixnum (* incx i)))))
				`(,one-arg-fn (aref x (+ offsetx (the fixnum (* incx i)))))))))
		  
		  (define-impl (,node-name :device LispTensor)
			       :save-for-backward (t nil)
			       :forward ((self x out)
					 (let ((caller (,caller-name (dtype x))))
					   `(,@(call-with-view
						#'(lambda (x-view o-view)
						    `(funcall
						      ,caller
						      (tensor-vec ,x)
						      (tensor-vec ,out)
						      ,(offset-of x-view 0)
						      ,(offset-of o-view 0)
						      ,(stride-of x-view 0)
						      ,(stride-of o-view 0)
						      ,(size-of x-view 0)))
						`(,x ,out))
					     ,out))))))))
  ;;===============|NodeName|=|Func|=|Dense?|=== 
  (define-math-impl AbsNode    abs    nil)
  (define-math-impl SignNode   signum nil)

  ;; Safety vs Speed?
  (define-math-impl SqrtNode sqrt t)
  (define-math-impl SquareNode #'(lambda (x) (* x x)) nil)

  
  (define-math-impl SinNode sin t)
  (define-math-impl CosNode cos t)
  (define-math-impl TanNode tan t)

  (define-math-impl ASinNode asin t)
  (define-math-impl ACosNode acos t)
  (define-math-impl ATanNode atan t)

  (define-math-impl SinhNode sinh t)
  (define-math-impl CoshNode cosh t)
  (define-math-impl TanhNode tanh t)

  ;;(define-math-impl ASinHNode asinh t)
  ;;(define-math-impl ACosHNode acosh t)
  ;;(define-math-impl ATanHNode atanh t)

  (define-math-impl ExpNode  exp t)
  (define-math-impl Log2Node  #'(lambda (x) (log x 2))  t)
  (define-math-impl Log10Node #'(lambda (x) (log x 10)) t)

  (define-math-impl LogENode log t)

  (define-math-impl Log1PNode #'(lambda (x) (log (1+ x))) t)
  )

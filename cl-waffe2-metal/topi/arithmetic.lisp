
(in-package :cl-waffe2/backends.metal)

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Add, Sub, Mul, Div, Move (+ - * / =)
;; ScalarAdd, ScalarSub, ScalarMul, ScalarDiv,
;; Reciprocal
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; Defines an BLAS-like element-wise metal kernel
;; TODO: add :cache-global option to avoid re-loading overheads
(macrolet ((def-kernel (op name dtype
			&aux (fname (symb name '- dtype)))
	     `(define-kernel (,fname :thread-position-in-grid id)
		  ;; X + Y -> Y
		  (void ((x* ,dtype :in)
			 (x-offset uint :in)
			 (incx     uint :in)
			 (y* ,dtype :io)
			 (y-offset uint :in)
			 (incy     uint :in)))
		(,op (aref y (+ y-offset (* incy id)))
		     (aref x (+ x-offset (* incx id))))))
	   (def (op name)
	     `(progn
		(def-metal-caller ,name (x y) (x x-offset incx y y-offset incy))
		,@(loop for case in *all-dtype-case*
			collect
			`(def-kernel ,op ,name ,case)))))
  (def setf  metal-move)
  (def incf  metal-add)
  (def decf  metal-sub)
  (def mulcf metal-mul)
  (def divcf metal-div))

(macrolet ((def-kernel (op name dtype
			&aux (fname (symb name '- dtype)))
	     `(define-kernel (,fname :thread-position-in-grid id)
		  ;; X + Y -> Y
		  (void ((x* ,dtype :io)
			 (x-offset uint :in)
			 (incx     uint :in)
			 (scal ,dtype :in)))
		(,op (aref x (+ x-offset (* incx id))) scal)))
	   (def (op name)
	     `(progn
		(def-metal-caller ,name (x) (x x-offset incx scal))
		,@(loop for case in *all-dtype-case*
			collect
			`(def-kernel ,op ,name ,case)))))
  (def incf  metal-add-scal)
  (def decf  metal-sub-scal)
  (def mulcf metal-mul-scal)
  (def divcf metal-div-scal))

(macrolet ((def-impl (node-name op)
	     `(define-impl
		  ;; A B -> A
		  (,node-name :device MetalTensor)
		  :forward ((self x y)
			    `(progn
			       ,(call-with-view
				 #'(lambda (x-view y-view)
				     `(,',op
				       ,(dtype x)
				       ,(size-of   x-view 0)
				       ,y
				       ,(offset-of y-view 0)
				       ,(stride-of y-view 0)
				       ,x
				       ,(offset-of x-view 0)
				       ,(stride-of x-view 0)))
				 (list x y))
			       ,x)))))
  (def-impl AddNode metal-add)
  (def-impl SubNode metal-sub)
  (def-impl MulNode metal-mul)
  (def-impl DivNode metal-div)
  (def-impl MoveTensorNode metal-move))

(macrolet ((def-impl (node-name op)
	     `(define-impl
		  ;; A B -> A
		  (,node-name :device MetalTensor)
		  :forward ((self x scalar)
			    `(progn
			       ,(call-with-view
				 #'(lambda (x-view)
				     `(,',op
				       ,(dtype x)
				       ,(size-of   x-view 0)
				       ,x
				       ,(offset-of x-view 0)
				       ,(stride-of x-view 0)
				       (tensor-vec ,scalar)))
				 (list x))
			       ,x)))))
  (def-impl ScalarAdd metal-add-scal)
  (def-impl ScalarSub metal-sub-scal)
  (def-impl ScalarMul metal-mul-scal)
  (def-impl ScalarDiv metal-div-scal))


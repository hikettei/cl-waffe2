
(in-package :cl-waffe2/backends.jit.lisp)

;;
;; delayed-node-impls.lisp provides define-impl forms of principle operations of JITLispTensor.
;;


;; Arithmetic operation family is originally declared as:
;; X <- op(X, Y)
(macrolet ((define-arith-impl (name lisp-op)
	     `(define-impl (,name
			    :device JITLispTensor
			    :extends (LispJIT-Blueprint))
			   :forward ((self x y)
				     (declare (ignore y))
				     (setf (blueprint-operand self) ',lisp-op)
				     `(progn ,x)))))
  (define-arith-impl AddNode +)
  (define-arith-impl SubNode -)
  (define-arith-impl MulNode *)
  (define-arith-impl DivNode /))


;; MoveTensor is declared as:
;; Move(A, B) -> A
;; A[~] B[~] -> A[~]
(define-impl (MoveTensorNode :device JITLispTensor :extends (LispJIT-Blueprint))
	     :forward ((self out target)
		       (declare (ignore target))
		       (progn
			 (setf (blueprint-operand self) 'move)
			 nil)
		       
		       `(progn ,out)))

(define-impl (InverseTensorNode :device JITLispTensor :extends (LispJIT-Blueprint))
	     :forward ((self x)
		       (setf (blueprint-operand self) 'inverse)
		       `(progn ,x)))


;;
;; Scalar-Mat Operation family are originally declared as:
;; (A[~] Scalar[scal] -> A[~] where scal = 1)
;;
(macrolet ((define-scalar-mat-impl (name lisp-op)
	     `(define-impl (,name
			    :device JITLispTensor
			    :extends (LispJIT-Blueprint))
			   :forward ((self A scalar)
				     (declare (ignore scalar))
				     (setf (blueprint-operand self) ',lisp-op)
				     `(progn ,A)))))
  
  (define-scalar-mat-impl ScalarAdd scalar-add)
  (define-scalar-mat-impl ScalarSub scalar-sub)
  (define-scalar-mat-impl ScalarMul scalar-mul)
  (define-scalar-mat-impl ScalarDiv scalar-div))


;; Todo: Element-wise kernels...



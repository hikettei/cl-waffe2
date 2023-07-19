
(in-package :cl-waffe2/backends.jit.lisp)

;;
;; delayed-node-impls.lisp provides define-impl forms of principle operations of JITLispTensor.
;;


;; Arithmetic operation family is originally declared as:
;; X <- op(X, Y)
(macrolet ((define-arith-impl (name lisp-op)
	     `(progn
		(define-impl (,name
			      :device JITLispTensor
			      :extends (LispJIT-Blueprint))
			     :forward ((self x y)
				       (declare (ignore y))
				       (setf (blueprint-opecode self) ',lisp-op)
				       `(progn ,x)))

		(defmethod implement-op ((opcode (eql ',lisp-op)) opAST &rest args)
		  `(,',lisp-op ,(car args) ,(second args))))))
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
			 (setf (blueprint-opecode self) 'move)
			 nil)
		       `(progn ,out)))

(defmethod implement-op ((opcode (eql 'move)) opAST &rest args)
  ;; A <- B
  `(setf ,(car args) ,(second args)))

(define-impl (InverseTensorNode :device JITLispTensor :extends (LispJIT-Blueprint))
	     :forward ((self x)
		       (setf (blueprint-opecode self) 'inverse)
		       `(progn ,x)))

(defmethod implement-op ((op (eql 'inverse)) opAST &rest args)
  `(/ 1 ,(car args)))
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
				     (setf (blueprint-opecode self) ',lisp-op)
				     `(progn ,A)))))
  
  (define-scalar-mat-impl ScalarAdd scalar-add)
  (define-scalar-mat-impl ScalarSub scalar-sub)
  (define-scalar-mat-impl ScalarMul scalar-mul)
  (define-scalar-mat-impl ScalarDiv scalar-div))


;; Todo: Element-wise kernels...



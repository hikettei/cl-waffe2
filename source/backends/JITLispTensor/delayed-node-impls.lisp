
(in-package :cl-waffe2/backends.jit.lisp)

;;
;; delayed-node-impls.lisp provides define-impl forms of principle operations of JITLispTensor.
;;

(defun only-when-no-grad (&rest inputs)
  (declare (ignore inputs))
  (not *no-grad*))


;; Arithmetic operation family is originally declared as:
;; X <- op(X, Y)
(macrolet ((define-arith-impl (name lisp-op)
	     `(progn
		(define-impl (,name
			      :device JITLispTensor
			      :reject-p #'only-when-no-grad
			      :extends (LispJIT-Blueprint))
			     :forward ((self x y)
				       (declare (ignore y))
				       (setf (blueprint-opecode self) ',lisp-op)
				       `(progn ,x)))

		(defmethod implement-op ((opcode (eql ',lisp-op)) opAST &rest args)
		  (make-iseq
		   `(,',lisp-op ,(car args) ,(second args))
		   (car args))))))
  (define-arith-impl AddNode +)
  (define-arith-impl SubNode -)
  (define-arith-impl MulNode *)
  (define-arith-impl DivNode /))


;; MoveTensor is declared as:
;; Move(A, B) -> A
;; A[~] B[~] -> A[~]
(define-impl (MoveTensorNode :device JITLispTensor :reject-p #'only-when-no-grad :extends (LispJIT-Blueprint))
	     :forward ((self out target)
		       (declare (ignore target))
		       (progn
			 (setf (blueprint-opecode self) 'move)
			 nil)
		       `(progn ,out)))

(defmethod implement-op ((opcode (eql 'move)) opAST &rest args)
  ;; A <- B
  (make-iseq (second args)
	     (car args)))

(define-impl (InverseTensorNode :device JITLispTensor :reject-p #'only-when-no-grad :extends (LispJIT-Blueprint))
	     :forward ((self x)
		       (setf (blueprint-opecode self) 'inverse)
		       `(progn ,x)))

(defmethod implement-op ((op (eql 'inverse)) opAST &rest args)
  (make-iseq `(/ 1 ,(car args)) (car args)))

;;
;; Scalar-Mat Operation family are originally declared as:
;; (A[~] Scalar[scal] -> A[~] where scal = 1)
;;
(macrolet ((define-scalar-mat-impl (name lisp-op)
	     `(define-impl (,name
			    :device JITLispTensor
			    :reject-p #'only-when-no-grad
			    :extends (LispJIT-Blueprint))
			   :forward ((self A scalar)
				     (declare (ignore scalar))
				     (progn
				       (setf (blueprint-opecode self) ',lisp-op)
				       nil)
				     `(progn ,A)))))
  (define-scalar-mat-impl ScalarAdd +)
  (define-scalar-mat-impl ScalarSub -)
  (define-scalar-mat-impl ScalarMul *)
  (define-scalar-mat-impl ScalarDiv /))


;; Todo: Element-wise kernels... (OK)
;; Todo: !sas-op (to reverse the order of ops)
;; Todo: Scalar And Scalar Kernel ... !sub uses
;; Todo: Eliminate move
;; Todo: Mathematical APIs
;; Todo: TEst




(in-package :cl-waffe2/backends.jit.lisp)

;;
;; delayed-node-impls.lisp provides define-impl forms of principle operations of JITLispTensor.
;;

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun only-when-no-grad (&rest inputs)
    (declare (ignore inputs))
    (not *no-grad*)))


;; Arithmetic operation family is originally declared as:
;; X <- op(X, Y)
(macrolet ((define-arith-impl (name lisp-op)
	     `(progn
		(define-impl (,name
			      :device JITLispTensor
			      :reject-p #'only-when-no-grad
			      :extends (LispJIT-Blueprint))
			     :forward ((self x y)
				       (progn
					 (setf (blueprint-use-var self) `(,x ,y))
					 (setf (blueprint-opecode self) ',lisp-op)
					 nil)
				       
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
		       (progn
			 (setf (blueprint-use-var self) `(,target))
			 (setf (blueprint-opecode self) 'move)
			 nil)
		       `(progn ,out)))

(defmethod implement-op ((opcode (eql 'move)) opAST &rest args)
  ;; A <- B
  (make-iseq (second args)
	     (car args)))

(define-impl (InverseTensorNode :device JITLispTensor :reject-p #'only-when-no-grad :extends (LispJIT-Blueprint))
	     :forward ((self x)
		       (setf (blueprint-use-var self) `(,x))
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
				     (progn
				       (setf (blueprint-use-var self) `(,A ,scalar))
				       (setf (blueprint-opecode self) ',lisp-op)
				       nil)
				     `(progn ,A)))))
  (define-scalar-mat-impl ScalarAdd +)
  (define-scalar-mat-impl ScalarSub -)
  (define-scalar-mat-impl ScalarMul *)
  (define-scalar-mat-impl ScalarDiv /))


;; Todo: Element-wise kernels... (OK)
;; Todo: Mathematical kernels

;; X[~] OUT[~] -> OUT[~]
(macrolet ((define-math-impl (node name &body impl)
	     `(progn
		(defmethod implement-op ((op (eql ',name)) opAST &rest inputs)
		  ,@(or impl
			`((make-iseq `(,',name ,(car inputs))
				     (second inputs)))))

		(define-impl (,node
			      :device JITLispTensor
			      :reject-p #'only-when-no-grad
			      :extends (LispJIT-Blueprint))
			     :forward ((self X out)
				       (declare (ignore out))
				       (progn
					 (setf (blueprint-use-var self) `(,X))
					 (setf (blueprint-opecode self) ',name)
					 nil)
				       `(progn ,X))))))
  (define-math-impl AbsNode abs)
  (define-math-impl SignNode signum)

  (define-math-impl SqrtNode sqrt)
  (define-math-impl SquareNode square
    (make-iseq `(* ,(car inputs) ,(car inputs))
	       (second inputs)))

  (define-math-impl SinNode sin)
  (define-math-impl ASinNode asin)
  (define-math-impl SinhNode sinh)
  (define-math-impl ASinHNode asinh)

  (define-math-impl CosNode cos)
  (define-math-impl AcosNode acos)
  (define-math-impl CoshNode cosh)
  (define-math-impl AcosHNode acosh)

  (define-math-impl TanNode tan)
  (define-math-impl ATanNode atan)
  (define-math-impl TanhNode tanh)
  (define-math-impl ATanHNode atanh)

  (define-math-impl ExpNode exp)
  (define-math-impl LogeNode log)

  (define-math-impl Log2Node log2
    (make-iseq `(log ,(car inputs) 2)
	       (second inputs)))

  (define-math-impl Log10Node log10
    (make-iseq `(log ,(car inputs) 10)
	       (second inputs))))


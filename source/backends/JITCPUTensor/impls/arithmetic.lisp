
(in-package :cl-waffe2/backends.jit.cpu)


;; ~~ List of arithmetic operations in cl-waffe2 ~~~~~~~~~~~~
;; Matrix and Matrix Operations:

;; AddNode/SubNode/MulNode/DivNode + InverseNode/MoveTensorNode



;; Arithmetic operation family is originally declared as:
;; X <- op(X, Y)

;; First, Try Solving the loop order
;; If the order is complicated, set reject-p=nil
;; And solve-loop-order ...

(macrolet ((define-arith-impl (name lisp-op op-name)
	     `(progn
		(define-impl (,name
			      :device JITCPUTensor
			      :extends (CPUJIT-Blueprint))
			     :forward ((self x y)
				       ;; Called at a Toplevel
				       (progn
					 (setf (blueprint-use-var self) `(,x ,y))
					 (setf (blueprint-opecode self) ',lisp-op)
					 nil)

				       ;; Embedding into JIT
				       `(progn ,x)))

		(defmethod translate-op ((opcode (eql ',lisp-op)) opAST &rest args)
		  (make-inst :modify
			     ,op-name
			     (car args)
			     (cdr args))))))
  (define-arith-impl AddNode + "+=")
  (define-arith-impl SubNode - "-=")
  (define-arith-impl MulNode * "*=")
  (define-arith-impl DivNode / "/="))

(define-impl (MoveTensorNode :device JITCPUTensor :extends (CPUJIT-Blueprint))
	     :forward ((self out target)
		       (progn
			 (setf (blueprint-use-var self) `(,out ,target))
			 (setf (blueprint-opecode self) 'move)
			 nil)
		       `(progn ,out)))

(define-impl (cl-waffe2/base-impl::MoveScalarTensorNode :device JITCPUScalarTensor :extends (CPUJIT-Scalar-Blueprint))
	     :forward ((self out target)
		       (progn
			 (setf (blueprint-use-var self) `(,out ,target))
			 (setf (blueprint-opecode self) 'move)
			 nil)
		       `(progn ,out)))

(defmethod translate-op ((opcode (eql 'move)) opAST &rest args)
  ;; A <- B
  (let ((self (tensor-backward (opAST-car opAST))))
    (if (movetensor-ignore-me self)
	(make-inst :set     "="   (car args) (cdr args))
	(make-inst :modify  "="   (car args) (cdr args)))))

(define-impl (InverseTensorNode :device JITCPUTensor :extends (CPUJIT-Blueprint))
	     :forward ((self x)
		       (progn
			 (setf (blueprint-use-var self) `(,x)
			       (blueprint-opecode self) 'inverse)
			 nil)
		       `(progn ,x)))

(defmethod translate-op ((opcode (eql 'inverse)) opAST &rest args)
  (make-inst :apply
	     (format nil "(~a)INV_SCALAR" (dtype->ctype (dtype (car args))))
	     (car args)
	     (list (car args))))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Scalar and Matrix Operations:
;; ScalarAdd/ScalarSub/ScalarMul/ScalarDiv
;;

(macrolet ((define-scalar-mat-impl (name lisp-op)
	     `(define-impl (,name
			    :device JITCPUTensor
			    :extends (CPUJIT-Blueprint))
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

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Scalar and Scalar Operations:
;; ScalarAndScalarAdd ScalarAndScalarSub ScalarAndScalarMul ScalarAndScalarDiv
;; MoveScalarTensorNode
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


(macrolet ((define-sas-op (name lisp-op)
	     `(define-impl (,name
			    :device JITCPUScalarTensor
			    :extends (CPUJIT-Scalar-Blueprint))
			   :forward ((self A scalar)
				     (progn
				       (setf (blueprint-use-var self) `(,A ,scalar))
				       (setf (blueprint-opecode self) ',lisp-op)
				       nil)
				     `(progn ,A)))))

  (define-sas-op cl-waffe2/base-impl::ScalarAndScalarAdd +)
  (define-sas-op cl-waffe2/base-impl::ScalarAndScalarSub -)
  (define-sas-op cl-waffe2/base-impl::ScalarAndScalarMul *)
  (define-sas-op cl-waffe2/base-impl::ScalarAndScalarDiv /))



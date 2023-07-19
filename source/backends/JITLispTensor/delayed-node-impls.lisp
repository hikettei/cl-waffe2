
(in-package :cl-waffe2/backends.jit.lisp)

;;
;; delayed-node-impls.lisp provides define-impl forms of principle operations of JITLispTensor.
;;


;; Arithmetic operation family is originally declared as:
;; X <- op(X, Y)
(macrolet ((define-arith-impl (name lisp-op save-for-backward)
	     `(define-impl (,name
			    :device JITLispTensor
			    :extends (LispJIT-Blueprint))
			   :save-for-backward ',save-for-backward
			   :forward ((self x y)
				     (declare (ignore y))
				     (setf (blueprint-op-func self) ',lisp-op)
				     `(progn ,x)))))
  (define-arith-impl AddNode + nil)
  (define-arith-impl SubNode - nil)
  (define-arith-impl MulNode * (t t))
  (define-arith-impl DivNode / (t t)))

;; A[~] B[~] -> A[~]
(define-impl (MoveTensorNode :device JITLispTensor :extends (LispJIT-Blueprint))
	     :forward ((self out target)
		       (declare (ignore target))
		       (setf (blueprint-op-func self) 'move)
		       `(progn ,out)))

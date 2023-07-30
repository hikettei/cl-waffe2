
(in-package :cl-waffe2/backends.jit.cpu)


;; Mathmatical (one-arg) functions are defined as:
;; X[~] OUT[~] -> OUT[~]
;;

(macrolet ((define-math-impl (node name c-func-name &key (cast nil))
	     `(progn
		(defmethod translate-op ((op (eql ',name)) opAST &rest inputs)
		  (make-inst :apply
			     (if ,cast
				 (format nil "(~a)~a"
					 (dtype->ctype (dtype (car inputs)))
					 ,c-func-name)
				 ,c-func-name)
			     (second inputs)
			     (list (car inputs))))

		(define-impl (,(symb 'scalar- node)
			      :device JITScalarTensor
			      :extends (CPUJIT-Scalar-Blueprint))
			     :forward ((self X out)
				       (declare (ignore out))
				       (progn
					 (setf (blueprint-use-var self) `(,X))
					 (setf (blueprint-opecode self) ',name)
					 nil)
				       `(progn ,X)))		

		(define-impl (,node
			      :device JITCPUTensor
			      :extends (CPUJIT-Blueprint))
			     :forward ((self X out)
				       (declare (ignore out))
				       (progn
					 (setf (blueprint-use-var self) `(,X))
					 (setf (blueprint-opecode self) ',name)
					 nil)
				       `(progn ,X))))))
  (define-math-impl AbsNode abs       "abs")
  (define-math-impl SignNode signum   "sign" :cast t)

  (define-math-impl SqrtNode sqrt     "sqrt")
  (define-math-impl SquareNode square "SQUARE_SCALAR")

  (define-math-impl SinNode sin       "sin")
  (define-math-impl ASinNode asin     "asin")
  (define-math-impl SinhNode sinh     "sinh")
  (define-math-impl ASinHNode asinh   "asinh")

  (define-math-impl CosNode cos       "cos")
  (define-math-impl AcosNode acos     "acos")
  (define-math-impl CoshNode cosh     "cosh")
  (define-math-impl AcosHNode acosh   "acosh")

  (define-math-impl TanNode tan       "tan")
  (define-math-impl ATanNode atan     "atan")
  (define-math-impl TanhNode tanh     "tanh")
  (define-math-impl ATanHNode atanh   "atanh")

  (define-math-impl ExpNode exp       "exp")
  (define-math-impl LogeNode log      "log")

  (define-math-impl Log2Node log2     "log2")
  (define-math-impl Log10Node log10   "log10"))


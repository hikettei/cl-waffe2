
(in-package :cl-waffe2/vm.nodes.test)

(defun M= (tensor1 tensor2)
  (every #'= (tensor-vec tensor1) (tensor-vec tensor2)))

;; composite-function -> composite-function Test

(defmodel (Multiply-Gradients (self)
	   :on-call-> ((self x grad)
		       (declare (ignore self))
		       (with-no-grad
			 (cl-waffe2/base-impl:A*=B X grad)))))

(defmodel-as (Multiply-Gradients)
  :where (X[~] Grad[~] -> X[~])
  :asif :function
  :named multiply-grads-static)

(defmodel (SinModel (self) :on-call-> ((self x) (declare (ignore self)) (cl-waffe2/base-impl:!sin x))))
(defmodel-as (SinModel) :where (X[~] -> Out[~]) :asif :node :named !sinmodel :differentiable t)

(defmodel-as (Multiply-Gradients)
  :where (X[~] Grad[~] -> out[~])
  :asif :node
  :named !mgrad
  :differentiable t)

;; :asif = :node/:function
;; Support: Multiple Arguments returning

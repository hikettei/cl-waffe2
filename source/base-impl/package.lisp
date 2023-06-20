
(in-package :cl-user)

(defpackage :cl-waffe2/base-impl
  (:use :cl
        :cl-waffe2/vm.generic-tensor
	:cl-waffe2/vm.nodes)
  (:export
   #:MoveTensorNode
   #:MoveScalarTensorNode
   #:movetensor-ignore-me
   #:movetensor-save-for-backward)
  (:export
   #:AddNode
   #:SubNode
   #:MulNode
   #:DivNode
   #:MatmulNode
   #:InverseTensorNode
   #:trans-a?
   #:trans-b?
   #:!matmul
   #:transposed-p
   #:!t)
  (:export
   #:!sum
   #:!mean
   #:!move
   #:!copy
   #:!copy-force
   #:!view
   #:!reshape
   #:!rankup
   #:!flexible
   #:proceed
   #:proceed-time
   #:proceed-backward)
  (:export
   :node-x
   :node-y))

(in-package :cl-waffe2/base-impl)
	
(defmacro with-export (name &body body)
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (export ',name)
     ,@body))

(defun extend-states (result extend-from)
  "the tensor returned keeps these status:
   flexible-p
   transposed-p"
  (setf (tensor-flexible-p result) (tensor-flexible-p extend-from))
  result)

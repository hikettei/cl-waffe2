
(in-package :cl-user)

(defpackage :cl-waffe2/base-impl
  (:use :cl
        :cl-waffe2/vm.generic-tensor
	:cl-waffe2/vm.nodes)
  (:export
   #:Lazy-Function-Node
   #:Lazy-Reduce-Node
   #:lazy
   #:lazy-reduce
   #:forward-of
   #:backward-of
   #:reduced-of)
  (:export
   #:~
   #:MoveTensorNode
   #:MoveScalarTensorNode
   #:movetensor-ignore-me
   #:mv-lazy-sv4bw
   #:movetensor-save-for-backward
   #:move-maybe-in-place)
  (:export
   #:AddNode
   #:SubNode
   #:MulNode
   #:DivNode
   #:ScalarMul
   #:ScalarAndScalarAdd
   #:ScalarAndScalarSub
   #:ScalarAndScalarMul
   #:ScalarAndScalarDiv
   #:MatmulNode
   #:InverseTensorNode
   #:trans-a?
   #:trans-b?
   #:!matmul
   #:transposed-p
   #:read-untransposed
   #:!t
   #:A+=B)
  (:export
   #:%transform
   #:->contiguous
   #:padding
   #:!sum
   #:!mean
   #:!move
   #:!copy
   #:!permute
   #:!view
   #:!reshape
   #:!rankup
   #:!flexible
   #:proceed
   #:proceed-time
   #:proceed-backward
   #:proceed-bench
   #:lazy-print
   #:logical-compiler-info)
  (:export
   :node-x
   :node-y)
  ;; Reshapers
  (:export
   #:broadcast-to
   #:rev-last-two
   #:torch-order))

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
  (setf (tensor-initial-offset result) (tensor-initial-offset extend-from))
  result)

(defun tensor-permuted-p (tensor)
  (not (equal (reverse (loop for i upfrom 0 below (dims tensor)
			     collect i))
	      (cl-waffe2/vm.generic-tensor::tensor-permute-order tensor))))



(in-package :cl-user)

(defpackage :cl-waffe2/base-impl
  (:use :cl
   :cl-waffe2/vm.generic-tensor
	:cl-waffe2/vm.nodes)
  (:export
   #:MoveTensorNode
   #:movetensor-ignore-me
   #:movetensor-save-for-backward)
  (:export
   #:AddNode
   #:SubNode
   #:MulNode
   #:DivNode
   #:MatmulNode
   #:!matmul)
  (:export
   #:!sum
   #:!move
   #:!copy
   #:!view)
  (:export
   :node-x
   :node-y))

(in-package :cl-waffe2/base-impl)
	
(defmacro with-export (name &body body)
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (export ',name)
     ,@body))


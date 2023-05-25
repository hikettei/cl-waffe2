
(in-package :cl-user)

(defpackage :cl-waffe2/base-impl
  (:use :cl
   :cl-waffe2/vm.generic-tensor
	:cl-waffe2/vm.nodes)
  (:export
   :AddNode
   :SubNode
   :MulNode
   :DivNode

   :node-out
   :node-x
   :node-y))

(in-package :cl-waffe2/base-impl)
	


(in-package :cl-user)

(defpackage :cl-waffe2/base-impl
  (:use :cl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes)
  (:export
   #:!sum)
  (:export
   :node-x
   :node-y))

(in-package :cl-waffe2/base-impl)
	

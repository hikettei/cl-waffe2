
(in-package :cl-user)

(defpackage :cl-waffe2/vm.nodes.generic-tensor
  (:use :cl
   :cl-waffe2/vm.generic-tensor :fiveam
   :cl-waffe2/base-impl
	:cl-waffe2/vm.nodes
	:cl-waffe2/backends.lisp))

(in-package :cl-waffe2/vm.nodes.generic-tensor)

(def-suite :test-tensor)
(in-suite :test-tensor)


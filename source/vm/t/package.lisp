
(in-package :cl-user)

(defpackage :cl-waffe2/vm.test
  (:use
   :cl
   :cl-waffe2/vm
   :cl-waffe2/vm.nodes
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/distributions
   :cl-waffe2/nn
   :fiveam
   :cl-waffe2/base-impl
   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.lisp
   :cl-waffe2/backends.jit.cpu))

(in-package :cl-waffe2/vm.test)

(def-suite :vm-test)
(in-suite  :vm-test)

(defun ~= (x y)
  (< (- x y) 0.00001))


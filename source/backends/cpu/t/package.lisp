
(in-package :cl-user)

(defpackage :cl-waffe2/backends.cpu.test
  (:use :cl
        :fiveam
        :cl-waffe2/base-impl
	:cl-waffe2/base-impl.test
        :cl-waffe2/backends.cpu
	:cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes))

(in-package :cl-waffe2/backends.cpu.test)

(def-suite :test-backends-cpu)
(in-suite :test-backends-cpu)


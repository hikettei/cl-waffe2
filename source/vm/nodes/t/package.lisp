
(in-package :cl-user)

(defpackage :cl-waffe2/vm.nodes.test
  (:use :cl
	:cl-waffe2 :cl-waffe2/nn
        :cl-waffe2/vm.generic-tensor
	:cl-waffe2/vm.nodes
        :cl-waffe2/backends.cpu
        :cl-waffe2/base-impl
        :fiveam
        :cl-waffe2/distributions))

(in-package :cl-waffe2/vm.nodes.test)

(def-suite :test-nodes)
(in-suite :test-nodes)


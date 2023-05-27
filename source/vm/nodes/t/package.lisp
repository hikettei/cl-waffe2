
(in-package :cl-user)

(defpackage :cl-waffe2/vm.nodes.test
  (:use :cl :cl-waffe2/vm.generic-tensor
	:cl-waffe2/vm.nodes
	:cl-waffe2/backends.cpu
	:fiveam))

(in-package :cl-waffe2/vm.nodes.test)

(def-suite :test-nodes)
(in-suite :test-nodes)


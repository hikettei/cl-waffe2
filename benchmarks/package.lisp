
(in-package :cl-user)

(defpackage :cl-waffe2/benchmark
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/nn
   :cl-waffe2/backends.jit.cpu
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl
   :cl-waffe2/distributions))

(in-package :cl-waffe2/benchmark)


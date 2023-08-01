
(in-package :cl-user)

(defpackage :cl-waffe2/backends.jit.cpu.test
  (:use :cl :cl-waffe2 :cl-waffe2/distributions :cl-waffe2/base-impl :cl-waffe2/vm.generic-tensor :cl-waffe2/vm.nodes :cl-waffe2/backends.cpu :cl-waffe2/backends.jit.cpu :fiveam :cl-waffe2/base-impl.test))

(in-package :cl-waffe2/backends.jit.cpu.test)

(def-suite :jit-cpu-test)

(in-suite :jit-cpu-test)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (add-tester JITCPUTensor)
  (sub-tester JITCPUTensor)
  (mul-tester JITCPUTensor)
  (div-tester JITCPUTensor)
  (move-tester JITCPUTensor)
  (scalar-add-tester JITCPUTensor JITCPUScalarTensor)
  (scalar-sub-tester JITCPUTensor JITCPUScalarTensor)
  (scalar-mul-tester JITCPUTensor JITCPUScalarTensor)
  (scalar-div-tester JITCPUTensor JITCPUScalarTensor)
  (sum-tester LispTensor JITCPUScalarTensor)
  (mathematical-test-set JITCPUTensor JITCPUScalarTensor))


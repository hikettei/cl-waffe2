
(in-package :cl-user)

(defpackage :cl-waffe2/backends.jit.cpu.test
  (:use :cl :cl-waffe2 :cl-waffe2/distributions :cl-waffe2/base-impl :cl-waffe2/vm.generic-tensor :cl-waffe2/vm.nodes :cl-waffe2/backends.cpu :cl-waffe2/backends.jit.cpu :fiveam :cl-waffe2/base-impl.test))

(in-package :cl-waffe2/backends.jit.cpu.test)

(def-suite :jit-cpu-test)

(in-suite :jit-cpu-test)

(defun doing-pool (a)
  (proceed (call (cl-waffe2/nn:AvgPool2D `(3 3)) a)))

(defun pool-test ()
  (let ((a (randn `(10 3 20 20))))
    (every #'=
	   (tensor-vec (doing-pool a))
	   (with-devices (JITCPUTensor CPUTensor cl-waffe2/backends.lisp:LispTensor)
	     (tensor-vec (doing-pool a))))))

(test unfold-avgpool2d-jit-test
  (is (pool-test)))


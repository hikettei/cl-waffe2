
(in-package :cl-waffe2/backends.jit.cpu)

(defclass JITCPUTensor (cl-waffe2/backends.cpu:CPUTensor) nil)

;; TODO: Rename -> JITCPUScalarTensor
(defclass JITScalarTensor (cl-waffe2/vm.generic-tensor:ScalarTensor) nil)

(deftype JITAbleTensors ()
  "JITAbleTensor is tensors which are subject to be compiled: JITCPUTensor and ScalarTensor."
  `(or JITCPUTensor cl-waffe2/vm.generic-tensor:ScalarTensor))

(defmacro with-cpu-jit ((&rest more-devices) &body body)
  `(with-devices (JITCPUTensor JITScalarTensor ,@more-devices)
     ,@body))



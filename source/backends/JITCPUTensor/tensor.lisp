
(in-package :cl-waffe2/backends.jit.cpu)

(defclass JITCPUTensor (cl-waffe2/backends.cpu:CPUTensor) nil)

(defclass JITCPUScalarTensor (cl-waffe2/vm.generic-tensor:ScalarTensor) nil)

(deftype JITAbleTensors ()
  "JITAbleTensor is tensors which are subject to be compiled: JITCPUTensor and ScalarTensor."
  `(or JITCPUTensor JITCPUScalarTensor))

(defmacro with-cpu-jit ((&rest more-devices) &body body)
  `(with-devices (JITCPUTensor JITCPUScalarTensor ,@more-devices)
     ,@body))



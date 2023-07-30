
(in-package :cl-waffe2/backends.jit.cpu)

(defclass JITCPUTensor (cl-waffe2/backends.cpu:CPUTensor) nil)

(deftype JITAbleTensors ()
  "JITAbleTensor is tensors which are subject to be compiled: JITCPUTensor and ScalarTensor."
  `(or JITCPUTensor cl-waffe2/vm.generic-tensor:ScalarTensor))


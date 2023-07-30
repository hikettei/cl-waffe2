
(in-package :cl-waffe2/backends.jit.cpu)

(defclass JITCPUTensor (cl-waffe2/backends.cpu:CPUTensor) nil)

(defclass JITCPUScalarTensor (cl-waffe2/vm.generic-tensor:ScalarTensor) nil)

(deftype JITAbleTensors ()
  "JITAbleTensor is tensors which are subject to be compiled: JITCPUTensor and ScalarTensor."
  `(or JITCPUTensor JITCPUScalarTensor))

(defmacro with-cpu-jit ((&rest more-devices) &body body)
  `(with-devices (JITCPUTensor JITCPUScalarTensor ,@more-devices)
     ,@body))


(declaim (inline tensor-ptr))
(defun tensor-ptr (tensor &key (offset 0))
  (declare (type JITCPUTensor tensor)
	   (type fixnum offset))
  #+sbcl
  (let ((ptr (sb-sys:vector-sap (sb-ext:array-storage-vector (the (simple-array * (*)) (tensor-vec tensor))))))
    (locally (declare (optimize (speed 1) (safety 1)))
      (cffi:incf-pointer ptr (the fixnum (* (the fixnum (cffi:foreign-type-size (dtype tensor))) offset)))))
  #-(or sbcl)
  (error "JITCPUTensor requires SBCL to access the storage vector!"))

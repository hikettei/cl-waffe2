
(in-package :cl-waffe2/backends.jit.cpu)

(defclass JITCPUTensor (cl-waffe2/backends.cpu:CPUTensor) nil)

(defclass JITCPUScalarTensor (cl-waffe2/vm.generic-tensor:ScalarTensor) nil)

(deftype JITAbleTensors ()
  "JITAbleTensor is tensors which are subject to be compiled: JITCPUTensor and ScalarTensor."
  `(or JITCPUTensor JITCPUScalarTensor))

(defun enable-cpu-jit-toplevel (&key
				  (compiler "gcc"))
  "
## [function] enable-cpu-jit-toplevel
"
  (setf *default-c-compiler* compiler)
  (cl-waffe2::set-devices-toplevel 'JITCPUTensor 'JITCPUScalarTensor)
  t)

(defmacro with-cpu-jit ((&rest more-devices) &body body)
  "
## [macro] with-cpu-jit
"
  `(with-devices (JITCPUTensor JITCPUScalarTensor ,@more-devices)
     ,@body))

;; Memo: https://groups.google.com/g/comp.lang.lisp/c/4aDbcVUBraQ
;; Pinning Arrays?
;; TODO: Do it outside call-with-view
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

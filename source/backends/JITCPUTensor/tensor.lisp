
(in-package :cl-waffe2/backends.jit.cpu)

(defclass JITCPUTensor (cl-waffe2/backends.cpu:CPUTensor) nil)

(defclass JITCPUScalarTensor (cl-waffe2/vm.generic-tensor:ScalarTensor) nil)

(deftype JITAbleTensors ()
  "JITAbleTensor is tensors which are subject to be compiled: JITCPUTensor and ScalarTensor."
  `(or JITCPUTensor JITCPUScalarTensor))

(defun enable-cpu-jit-toplevel (&key
				  (more-devices)
				  (compiler "gcc")
				  (viz-compiled-code nil)
				  (flags '("-fPIC" "-O3" "-march=native")))
  "
## [function] enable-cpu-jit-toplevel

```lisp
(enable-cpu-jit-toplevel (&key
			  (more-devices)
			  (compiler \"gcc\")
			  (viz-compiled-code nil)
			  (flags '(\"-fPIC\" \"-O3\" \"-march=native\"))))
```

Sets `JITCPUTensor` and `JITCPUScalarTensor` to the priority of backends. Place this function in the top of your code that JIT Compiling is needed. Of course, `JITCPUTensor` is developed as a one of `external backends` in cl-waffe2, therefore Local JIT compilation with the `with-devices` macro is another valid option.

### Inputs

`more-devices[List]` specify the list of device names. they have lower priority than `JITCPUTensor`

`viz-compiled-code[boolean]` Set t to display the compiled c codes.
"
  (setf *default-c-compiler* compiler
	*viz-compiled-code* viz-compiled-code
	*compiler-flags* flags)
  (apply #'cl-waffe2:set-devices-toplevel 'JITCPUTensor 'JITCPUScalarTensor more-devices)
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

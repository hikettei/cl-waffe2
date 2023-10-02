
(in-package :cl-waffe2/backends.jit.cpu)

(defclass JITCPUTensor (cl-waffe2/backends.cpu:CPUTensor) nil
  (:documentation "
## [AbstractTensor] JITCPUTensor
"))

(defmethod current-backend-state ((backend-name (eql 'JITCPUTensor)))
  (format nil "compiler=~a flags=~a viz=~a"
	  *default-c-compiler*
	  *compiler-flags*
	  *viz-compiled-code*))

(deftype JITAbleTensors ()
  "JITAbleTensor is tensors which are subject to be compiled: JITCPUTensor and ScalarTensor."
  `(or JITCPUTensor))

(defun cpujit-set-config (&key
			    (compiler "gcc")
			    (viz-compiled-code nil)
			    (openmp nil)
			    (flags '("-fPIC" "-O3" "-march=native")))
  "
## [function] cpujit-set-config

```lisp
(cpujit-set-config (&key
			  (compiler \"gcc\")
			  (viz-compiled-code nil)
                          (openmp nil)
			  (flags '(\"-fPIC\" \"-O3\" \"-march=native\"))))
```

(TODO cuz 腱鞘炎なった)
 
### Inputs

"
  (setf *default-c-compiler* compiler
	*viz-compiled-code* viz-compiled-code
	*use-open-mp* openMP
	*compiler-flags* flags)
  t)

(defmacro with-cpu-jit ((&rest more-devices) &body body)
  "
## [macro] with-cpu-jit

Under this macro, two backends (`JITCPUTensor` and `JITCPUScalarTensor`) are installed at the top of the priority list.
"
  `(with-devices (JITCPUTensor ,@more-devices)
     ,@body))

(defmacro with-tensor-ptr ((bind tensor) &body body)
  `(progn
     ;; Ensure that tensor storage vector has allocated.
     (tensor-vec ,tensor)
     (cffi:with-pointer-to-vector-data (,bind (cl-waffe2/vm.generic-tensor::vec ,tensor))
       (declare (type cffi-sys:foreign-pointer ,bind))
       ,@body)))

(defmacro with-tensor-ptrs ((&rest input-forms) &body body)
  (labels ((expand (rest-forms)
	     (if rest-forms
		 `(with-tensor-ptr (,(caar rest-forms) ,(second (car rest-forms)))
		    ,(expand (cdr rest-forms)))
		 `(progn ,@body))))
    (expand input-forms)))


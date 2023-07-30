
(in-package :cl-waffe2/backends.jit.cpu)

;; Ref: https://stackoverflow.com/questions/6612169/compile-a-stream-of-data-in-c
(defun load-foreign-function (source
			      &key
				;; (disassemble t)
				(compiler "gcc")
				(lang "c"))
  (declare (type string source compiler))

  (uiop:with-temporary-file (:pathname sharedlib :type "so" :keep t)
    :close-stream
    (let* ((cmd
	     ;; gcc -shared -o sharedlib
	     (list
	      compiler "-shared"
	      "-x" lang
	      "-o" (uiop:native-namestring sharedlib) "-"))
	   (process-info (uiop:launch-program
			  cmd
			  :input :stream
			  :error-output :stream))
	   (input (uiop:process-info-input process-info))
	   (error-output (uiop:process-info-error-output process-info)))
      (unwind-protect (princ source input)
	(close input))
      (unless (zerop (uiop:wait-process process-info))
	(error "cl-waffe2/backends.jit.cpu: Failed to compile a shared library:~%~a~%"
	       (alexandria:read-stream-content-into-string error-output))))
    (cffi:load-foreign-library sharedlib)))

(defun expand-funcall-form (function-name args views)
  (declare (type string function-name)
	   (type list args views))

  (let ((view-count 0))
    (flet ((read-view (nth)
	     (let ((out (nth nth views)))
	       out)))
      `(cffi:foreign-funcall
	,function-name
	,@(if (null views)
	      `(:uint32 0)
	      `(:uint32 ,(size-of (car views) 0)))
	,@(loop for arg in args
		append
		(typecase arg
		  (JITCPUScalarTensor
		   ;; pass the pointer directly
		   ;; sb-ext:int-sap
		   `(,(dtype arg) (tensor-vec ,arg)))
		  (JITCPUTensor
		   (prog1
		       ;; tensor_ptr tensor_stride ...
		       ;; (dtype val)
		       `(:pointer
			 (tensor-ptr ,arg :offset ,(offset-of (read-view view-count) 0))
			 :int32;;(:pointer :int32)
			 ,(stride-of (read-view view-count) 0))
		     (incf view-count)))
		  (T (error "unknown type of arguments: ~a" arg))))
	:void))))


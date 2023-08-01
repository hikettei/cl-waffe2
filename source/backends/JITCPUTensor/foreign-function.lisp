
(in-package :cl-waffe2/backends.jit.cpu)

(defparameter *default-c-compiler* "gcc" "
## [parameter] `*default-c-compiler*`

Specify the command to compile the generated c codes. In default, \"gcc\".
")

(defparameter *compiler-flags* '("-fPIC" "-O3" "-march=native") "
## [parameter] `*compiler-flags*`

In default, `*compielr-flags*` = `'(\"-fPIC\" \"-O3\" \"-march=native\")`
")

(defparameter *viz-compiled-code* nil "
## [parameter] `*viz-compiled-code*`

Set t to display the compiled c code to terminal. In default, `nil`
")

;; Ref: https://stackoverflow.com/questions/6612169/compile-a-stream-of-data-in-c
(defun load-foreign-function (source
			      &key
				(compiler *default-c-compiler*)
				(lang "c"))
  (declare (type string source compiler))

  (uiop:with-temporary-file (:pathname sharedlib :type "so" :keep t)
    :close-stream
    (let* ((cmd
	     ;; gcc -shared -o sharedlib
	     (append
	      (list
	       compiler "-shared"
	       "-x" lang)
	      *compiler-flags*
	      (list "-o" (uiop:native-namestring sharedlib) "-")))
	   (process-info (uiop:launch-program
			  cmd
			  :input :stream
			  :error-output :stream))
	   (input (uiop:process-info-input process-info))
	   (error-output (uiop:process-info-error-output process-info)))
      (unwind-protect (princ source input)
	(close input))
      (unless (zerop (uiop:wait-process process-info))
	(error "cl-waffe2/backends.jit.cpu: Failed to compile a shared library:~%~a~%

Compiled with: ~a
Tips: Modify cl-waffe2/backends.jit.cpu:*default-c-compiler* to switch compilers for cl-waffe2 to use. (If the error is related to gcc.)"
	       (alexandria:read-stream-content-into-string error-output)
	       (with-output-to-string (out)
		 (dolist (c cmd) (princ c out) (princ " " out))))))
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
		   `(:pointer ,(int-sap-id arg)))
		  (JITCPUTensor
		   (prog1
		       ;; tensor_ptr tensor_stride ...
		       `(:pointer
			 (tensor-ptr (read-result ,arg) :offset ,(offset-of (read-view view-count) 0))
			 :int32
			 ,(stride-of (read-view view-count) 0))
		     (incf view-count)))
		  (T (error "unknown type of arguments: ~a" arg))))
	:void))))


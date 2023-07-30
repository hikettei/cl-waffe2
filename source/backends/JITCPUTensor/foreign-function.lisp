
(in-package :cl-waffe2/backends.jit.cpu)

;; Ref: https://stackoverflow.com/questions/6612169/compile-a-stream-of-data-in-c
(defun load-foreign-function (source
			      &key
				;; (disassemble t)
				(compiler "gcc-13")
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
    (print sharedlib)
    ;;(cffi:load-foreign-library "")
    ))

;; (defmacro call-jit-function

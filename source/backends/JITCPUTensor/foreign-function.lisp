
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

(defun jit-form-init! (jitf)
  (setf (jit-caller-form jitf) (compile nil (jit-funcall-form jitf))))

(defun jit-funcall (jitf &rest args)
  (apply (the function (jit-caller-form jitf)) args))

(defun jit-funcall-form (jit-compiled-kernel)
  (with-slots ((name name) (dynamic-symbols dynamic-symbols) (args args)) jit-compiled-kernel
    `(lambda (,@(map 'list #'tensor-id args))
       (with-tensor-ptrs (,@(loop for arg in args
				  collect `(,(cPointer arg) ,(tensor-id arg))))
	 (cffi:foreign-funcall
	  ,name
	  ,@(loop for symbol in dynamic-symbols
		  append
		  `(:uint32 (cl-waffe2/vm:maybe-observe-axis ',symbol)))
	  ,@(loop for arg in args
		  append
		  (append
		   `(:pointer ,(cPointer arg))))
	  :void)))))


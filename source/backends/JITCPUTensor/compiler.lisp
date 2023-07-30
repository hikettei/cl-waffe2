
(in-package :cl-waffe2/backends.jit.cpu)

(defvar *compiled-code-buffer* nil "The variable collects generated C codes.")
(defparameter *indent-width* 0)

(defmacro with-indent (indent &body body)
  "Add indentations to generated C code with write-c-line"
  `(let ((*indent-width* ,indent)) ,@body))

(defmacro with-compiling-mode (&body body)
  "Initializes *compiled-code-buffer*"
  `(with-output-to-string (*compiled-code-buffer*)
     ,@body))

(defun write-buff (control-string &rest args)
  "Appends the given characters to *compiled-code-buffer*."
  (apply #'format *compiled-code-buffer* control-string args))

(defun write-c-line (control-string &rest args)
  "Appends the given characters to *compiled-code-buffer*."
  (dotimes (i *indent-width*) (princ " " *compiled-code-buffer*))
  (apply #'format *compiled-code-buffer* control-string args))

(defparameter *includes*
  `("immintrin.h" "stdbool.h" "stdlib.h" "math.h" "stdio.h" "stdint.h")
  "An list of headers that generated code loads first.")

(defun place-toplevel-form (cffi-call-name tensors)
  "Places headers, function definition and macros."
  (write-buff "~%~%#pragma simd~%")
  ;; #pragma GCC optimize ("O3")
  ;; #pragma GCC target "avx2" avx512 ...
  ;; ^ (TODO) Identify CPU by cpu_has_avx512 ...
  
  (loop for include in *includes*
	do (write-buff "#include <~a>~%" include))

  (write-buff "~%~a;~%~%" (apply #'cFunction cffi-call-name tensors))

  ;; Utils
  (write-buff "#define INV_SCALAR(scal) 1 / scal;~%~%")
  (write-buff "#define SQUARE_SCALAR(scal) scal * scal;~%~%")
  )

(defun cAref (tensor)
  "Reading the given tensor's id, the function returns a string which corresponds to aref in C"
  (declare (type AbstractTensor tensor))
  (if (typep tensor 'JITCPUTensor)
      (format nil "~a[i * ~a_STRIDE]"
	      (tensor-id tensor)
	      (tensor-id tensor))
      (format nil "~a" (tensor-id tensor))))

(defun cFunction (function-name &rest arguments)
  "Header:
void function-name (int size, float * restrict x1, int stride, int offset, float* x2 ...)

  Returns the definition form of given function."

  (let ((arguments-form
	  (with-compiling-mode
	    (write-buff "(uint32_t restrict * size, ")
	    (loop for arg in arguments
		  for n upfrom 0
		  do (cVar arg
			   :restrict (= 1 (count (tensor-id arg) arguments :key #'tensor-id))
			   :comma (or
				   (typep arg 'JITCPUTensor)
				   (not (= n (1- (length arguments))))))
		  if (typep arg 'JITCPUTensor)
		    do (cStride arg :comma (not (= n (1- (length arguments))))))
	    (write-buff ")"))))
    (format nil "void ~a~a~%" function-name arguments-form)))

(defun invoke-compiler! (function-name toplevel)
  "
Recursively exploring the computation node staring from toplevel, the function invoke-compiler! appends C codes depending on translate-op method to the current buffer.

Return: (values envolved-tensors(but ScalarTensor) toplevel)
"
  (declare (type JITAbleTensors toplevel))

  (let* ((*compiled-tensors* `(,toplevel))
	 (envolved-nodes (confirm-compiling-area toplevel))
	 (function-form  (apply #'cFunction function-name *compiled-tensors*)))
    (values
     ;; Used for expanding call-with-view
     (loop for tensor in *compiled-tensors*
	   if (typep tensor 'JITCPUTensor)
	     collect tensor)
     (with-compiling-mode
       (place-toplevel-form function-name *compiled-tensors*)

       ;; void function-name (...) { ...
       (write-buff "~a { ~%" function-form)
       (with-indent 4
	 (write-c-line "for(int i=0; i<size; i++) {~%")
	 (with-indent 8
	   (ir->C envolved-nodes))
	 (write-c-line "}~%"))
       (write-c-line "}~%")))))


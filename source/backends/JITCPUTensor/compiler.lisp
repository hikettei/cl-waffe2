
(in-package :cl-waffe2/backends.jit.cpu)

(defvar *compiled-code-buffer* nil "The variable collects generated C codes.")
(defparameter *indent-width* 0)

(defmacro with-indent (indent &body body)
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
  `("immintrin.h" "stdbool.h" "math.h" "stdio.h" "stdint.h"))

(defun place-toplevel-form (cffi-call-name tensors)
  (write-buff "~%~%#pragma simd~%")
  ;; #pragma GCC optimize ("O3")
  ;; #pragma GCC target "avx2" avx512 ...
  ;; ^ (TODO) Identify CPU by cpu_has_avx512 ...
  
  (loop for include in *includes*
	do (write-buff "#include <~a>~%" include))

  (write-buff "~%~a;~%~%" (apply #'cFunction cffi-call-name tensors)))

(defun cAref (tensor)
  (declare (type AbstractTensor tensor))
  (format nil "~a[i * ~a_STRIDE]"
	  (tensor-id tensor)
	  (tensor-id tensor)))

;; Tensor -> (Tensor-vec stride offset)
(defun cFunction (function-name &rest arguments)
  "Header:
void function-name (int size, float * restrict x1, int stride, int offset, float* x2 ...)"

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
    (format nil "void ~a~a;~%" function-name arguments-form)))

(defun invoke-compiler! (function-name toplevel)
  "
Return: (values envolved-tensors(but ScalarTensor) toplevel)
"
  (declare (type JITAbleTensors toplevel))

  (let* ((*compiled-tensors* `(,toplevel))
	 (envolved-nodes (confirm-compiling-area toplevel))
	 (function-form  (apply #'cFunction function-name *compiled-tensors*)))
    (values
     *compiled-tensors*
     (with-compiling-mode
       (place-toplevel-form function-name *compiled-tensors*)

       ;; void function-name (...) { ...
       (write-buff "~a { ~%" function-form)
       (with-indent 4
	 (write-c-line "for(int i=0;i<size;i++) {~%")
	 (with-indent 8
	   (ir->C envolved-nodes))
	 (write-c-line "}~%"))
       (write-c-line "}~%")))))


;; Workload:
;; 1. CPUのAVX拡張命令を特定
;; 2. コンパイルの設定
;; 3. gccの設定
;; 4. 動的にCFFIから読み込む



(in-package :cl-waffe2/backends.jit.cpu)

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Generating a C Code from cl-waffe2.
;; The scope of JIT: Whether the generated code can be expressed with only one `for`.
;; 
;; 



;; Memo: OffsetはLisp側で加算することにする
;; Pragma SIMDで自動ベクトル化
;; gemmはOpenBLASのAPIを呼び出す
;; void NAME ...

;; 

(defparameter *includes*
  `("immintrin.h" "stdbool.h" "math.h" "stdio.h" "stdint.h"))

(defun place-toplevel-form (cffi-call-name tensors)
  (write-buff "~%~%#pragma simd~%")
  ;; #pragma GCC optimize ("O3")
  ;; #pragma GCC target "avx2" avx512 ...
  ;; ^ (TODO) cpu_has_avx512 ...
  
  (loop for include in *includes*
	do (write-buff "#include <~a>~%" include))

  (write-buff "~%~a;~%~%" (apply #'cFunction cffi-call-name tensors)))

;; Tensor -> (Tensor-vec stride offset)
(defun cFunction (function-name &rest arguments)
  "Header:
void function-name (int size, float * restrict x1, int stride, int offset, float* x2 ...)"

  (let ((arguments-form
	  (with-compiling-mode
	    (write-buff "(uint32_t size, ")
	    (loop for arg in arguments
		  for n upfrom 0
		  do (cVar arg
			   :restrict (= 1 (count (tensor-id arg) arguments :key #'tensor-id))
			   :comma (not (= n (1- (length arguments)))))
		  if (typep arg 'JITCPUTensor)
		    do (cStride arg :comma (not (= n (1- (length arguments))))))
	    (write-buff ")"))))
    (format nil "void ~a~a" function-name arguments-form)))

;; Dynamically calls compiled c shared lib <-> Lisp Programs via CFFI.

;; (defun call-c-form (tensors))


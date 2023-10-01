
(in-package :cl-waffe2/backends.jit.cpu)

(defvar *compiled-code-buffer* nil "The variable collects generated C codes.")
(defparameter *indent-width* 0)

(defparameter *use-open-mp* nil)

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

  (when (null *caching-c-source*) 
    (write-buff "~%#pragma SIMD~%")
    ;;(write-buff "#pragma GCC optimize (\"O3\")~%")
    ;;(write-buff "#pragma GCC target \"avx2\"")
    ;; #pragma GCC target "avx2" avx512 ...
    
    (loop for include in *includes*
	  do (write-buff "#include <~a>~%" include))

    (when *use-open-mp*
      (write-buff "#include <omp.h>~%"))

    (write-buff "~%~a;~%~%" (apply #'cFunction cffi-call-name tensors))

    ;; Utils
    (write-buff "#define INV_SCALAR(scal) 1 / scal;~%~%")
    (write-buff "#define SQUARE_SCALAR(scal) scal * scal;~%~%")
    ))

(defun cAref (tensor &key (pointer nil))
  "Reading the given tensor's id, the function returns a string which corresponds to aref in C"
  (declare (type AbstractTensor tensor))
  (if (typep tensor 'JITCPUTensor)
      (format nil "~a[i * ~a_STRIDE]"
	      (tensor-id tensor)
	      (tensor-id tensor))
      (if pointer
	  (format nil "*~a" (tensor-id tensor))
	  (format nil "~a"    (tensor-id tensor)))))

(defun cFunction (function-name &rest arguments)
  "Header:
void function-name (int size, float * restrict x1, int stride, int offset, float* x2 ...)

  Returns the definition form of given function."

  (let ((arguments-form
	  (with-compiling-mode
	    (write-buff "(uint32_t size, ")
	    (loop for arg in arguments
		  for n upfrom 0
		  do (cVar arg
			   :restrict (= 1 (count (tensor-id arg) arguments :key #'tensor-id))
			   :comma (or
				   (typep arg 'JITCPUTensor)
				   (not (= n (1- (length arguments)))))
			   :pointer t)
		  if (typep arg 'JITCPUTensor)
		    do (cStride arg :comma (not (= n (1- (length arguments))))))
	    (write-buff ")"))))
    (format nil "void ~a~a" function-name arguments-form)))

(defun insert-loop-for ()
  (let ((back-indent-size (max 0 (- *indent-width* 4))))
    (with-indent back-indent-size
      (write-c-line "}~%~%")
      (when *use-open-mp*
	(write-c-line "#pragma omp parallel for~%"))
      (write-c-line "for(int i=0; i<size; i++) {~%"))))

(defun iterator-symbols (rank)
  (loop for r upfrom 0 below rank
	collect
	(format nil "~a~a" (gensym "L") (code-char (+ 65 r)))))

;; [TODO] element-wise op fusion
;; [TODO] Use SLEEF Backend For Mathematical Kernel
;; [TODO] AVXnnn Intrinsics?
;; [TODO] Adjustable Shape Support
;; [TODO] Runtime Mode ... able to include adjustable shape
;; AbstractLoop To C Compiler
(defun invoke-compiler! (function-name toplevel)
  "
Recursively exploring the computation node staring from toplevel, the function invoke-compiler! appends C codes depending on translate-op method to the current buffer.

Return: (values arguments envolved-tensors(but ScalarTensor) scalars toplevel)
"
  (declare (type JITAbleTensors toplevel))

  ;; solve-loop-order:
  ;;  Creates an blueprint of optimized loop order
  ;;  This compiler basically follows its instruction, generating corresponding loops in C.
  (let* ((abstract-loop (solve-loop-order (tensor-variables toplevel) 1 nil :mode :runtime))
	 (adjustable-shape (loop for loop in abstract-loop
				 if (symbolp (aloop-size loop))
				   collect (aloop-size loop))))
    (print adjustable-shape)

    (with-compiling-mode
      ;; [TODO]
      ;; 引数 ... Tensorのポインタ + Adjustable Shape (IF ANY)
      (place-toplevel-form function-name (tensor-variables toplevel))

      (write-c-line "~a { ~%" (cFunction function-name toplevel))
      (with-indent 4
	(loop with indices list = (iterator-symbols (length abstract-loop))
	      for *indent-width* upfrom 4 by 4
	      for index-char in indices
	      for loop        in abstract-loop do
		(print loop)
		(case (aloop-mode loop)
		  (:batch
		   ;; If *use-open-mp* is set to T and the currently processing loop is the first one
		   ;; Inserts the pragma:
		   (when (and (= *indent-width* 4)
			      *use-open-mp*)
		     (write-c-line "#pragma omp parallel for~%"))
		   (write-c-line
		    "for (int ~a=0;~a<~a;~a++) {~%"
		    index-char
		    index-char
		    (aloop-size loop)
		    index-char))
		  (T
		   ;; Excepted one of: :apply :apply-flatten
		   (when (and (= *indent-width* 4)
			      *use-open-mp*)
		     (write-c-line "#pragma omp parallel for ~%"))
		   
		   (write-c-line
		    "for (int ~a=0;~a<~a;~a++) {~%"
		    index-char
		    index-char
		    (aloop-element-n loop)
		    index-char)

		   (let ((*indent-width* (+ 4 *indent-width*)))
		     ;; [TODO]
		     (print (aloop-by loop))
		     (write-c-line
		      "~a[XXX] = ~a[XXX];~%"
		      (tensor-id toplevel)
			(tensor-id toplevel))))))


	;; Closing Brackets
	(loop for *indent-width* downfrom (* 4 (length abstract-loop)) to 0 by 4 do
	  (write-c-line "}~%"))))))

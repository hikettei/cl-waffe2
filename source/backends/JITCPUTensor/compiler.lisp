
(in-package :cl-waffe2/backends.jit.cpu)


;; ~~ Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(defvar *compiled-code-buffer* nil
  "A parameter temporary storing the result of generated C code, being initialized with (with-compiling-mode ...) macro.")

(defparameter *indent-width* 0
  "A parameter indicating the width of indentation, effecting on write-c-line function")

(defparameter *use-open-mp* nil
  "A parameter indicating whether using OpenMP or not. In default set to T.
   This parameter is modified via enable-jit-cpu function.")

(defmacro with-indent (indent &body body)
  "Add indentations to generated C code with write-c-line"
  `(let ((*indent-width* ,indent)) ,@body))

(defmacro with-compiling-mode (&body body)
  "Initializes *compiled-code-buffer*"
  `(with-output-to-string (*compiled-code-buffer*)
     ,@body))
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; ~~ IO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(defun write-buff (control-string &rest args)
  "Appends the given characters to *compiled-code-buffer* without indentations"
  (apply #'format *compiled-code-buffer* control-string args))

(defun write-c-line (control-string &rest args)
  "Appends the given characters to *compiled-code-buffer*."
  (dotimes (i *indent-width*) (princ " " *compiled-code-buffer*))
  (apply #'format *compiled-code-buffer* control-string args))
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; Node -> Compiling Parts:

(defparameter *includes*
  `("immintrin.h" "stdbool.h" "stdlib.h" "math.h" "stdio.h" "stdint.h")
  "A list of header files envolved in compiling.")

(defun place-toplevel-form (cffi-call-name tensors)
  "Places headers, function definition and macros."

  ;; Ensures that this call is the first time.
  ;; Since only required once when compiling.
  (when (null *caching-c-source*) 
    (write-buff "~%#pragma SIMD~%")
    (write-buff "#pragma GCC optimize (\"O3\")~%")
    
    (loop for include in *includes* do
      (write-buff "#include <~a>~%" include))

    (when *use-open-mp*
      (write-buff "#include <omp.h>~%"))

    (write-buff "~%~a;~%~%" (apply #'cFunction cffi-call-name tensors))))

(defun iterator-symbols (rank)
  (loop for r upfrom 0 below rank
	collect
	(format nil "~a~a" (gensym "L") (code-char (+ 65 r)))))

(defstruct (JIT-Compiled-Kernel
	    (:conc-name jit-))
  (name "" :type string)
  (args nil :type list)
  (dynamic-symbols nil :type list)
  (body "" :type string)
  (caller-form #'(lambda ()) :type function))

(defmethod print-object ((obj JIT-Compiled-Kernel) stream)
  (format stream "<JITCompiledKernel:~a~a ~a>"
	  (jit-name obj)
	  (or (jit-dynamic-symbols obj) "")
	  (apply #'concatenate 'string
		 (butlast
		  (loop for arg in (jit-args obj)
			append
			(list (tensor-id arg) " "))))))
	    
;; [TODO] element-wise op fusion
;; [TODO] Use SLEEF Backend For Mathematical Kernel
;; [TODO] AVXnnn Intrinsics?
;; [TODO] Adjustable Shape Support
;; [TODO] Runtime Mode ... able to include adjustable shape
;; AbstractLoop To C Compiler
;; defpath関連削除

(defun generate-c-kernel (function-name variables abstract-loop instructions)
  (with-compiling-mode
    ;; place-toplevel-form appends these forms:
    ;;  - includes
    ;;  - header
    ;;  - macros
    (place-toplevel-form function-name variables)
    (write-c-line "~a { ~%" (cFunction function-name variables))
    (with-indent 4
      (loop with indices list = (iterator-symbols (length abstract-loop))
	    for *indent-width* upfrom 4 by 4
	    for index-char in indices
	    for loop        in abstract-loop do
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
		   (dolist (inst instructions)
		     (render-instruction inst indices)))))))

    ;; Closing Brackets
    (loop for *indent-width* downfrom (* 4 (length abstract-loop)) to 0 by 4 do
      (write-c-line "}~%"))))

(defun invoke-compiler! (function-name instructions)
  "Compiles to C Kernel.

kernel = instructions[last](... instructions[1](instructions[0](Arguments)))

Inputs:
 - function-name[symbol]
 - instructions a list of Instruction

Return:
 JIT-Compiled-Kernel"
  (declare (type list instructions))

  ;; solve-loop-order:
  ;;  Creates an blueprint of optimized loop order
  ;;  This compiler basically follows its instruction, generating corresponding loops in C.
  (let* ((variables (collect-variables instructions))
	 (abstract-loop (solve-loop-order variables 1 nil :mode :runtime))
	 (adjustable-shape (loop for loop in abstract-loop
				 if (symbolp (aloop-size loop))
				   collect (aloop-size loop))))
    (make-jit-compiled-kernel
     :name             function-name
     :args             variables
     :dynamic-symbols  adjustable-shape
     :body             (generate-c-kernel function-name variables abstract-loop instructions))))



(in-package :cl-waffe2/backends.jit.cpu)

;; [TODO]
;; With enough time, we could introduce these features to a JIT Compiler:
;;  - Polyhedral Compiler
;;  - Tiling, Unrolling
;;  - considering L1/L2 caching, reducing

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

(defun place-toplevel-form (cffi-call-name shapes tensors)
  "Places headers, function definition and macros."

  ;; Ensures that this call is the first time.
  ;; Since only required once when compiling.
  (when (string= *lazy-c-source* "")
    (write-buff "~%#pragma simd~%")
    ;;(write-buff "#pragma GCC optimize (\"O3\")~%")
    
    (loop for include in *includes* do
      (write-buff "#include <~a>~%" include))

    (when *use-open-mp*
      (write-buff "#include <omp.h>~%")))
  
  (write-buff "~%~a;~%~%" (cFunction cffi-call-name shapes tensors)))

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
  (format stream "<JITCompiledKernel:~a~a ~a{
~a
}>"
	  (jit-name obj)
	  (or (jit-dynamic-symbols obj) "")
	  (apply #'concatenate 'string
		 (butlast
		  (loop for arg in (jit-args obj)
			append
			(list (format nil "~a" (tensor-id arg)) " "))))
	  (jit-body obj)))

(defun pragma-omp (make-me-private)
  (format
   nil
   "#pragma omp parallel for~%"
   #|
   (apply
    #'concatenate
    'string
    (butlast
     (loop for var in make-me-private
	   append
   (list var ","))))
   |#
   ))

(defstruct (Iteration
	    (:constructor
		make-iteration (rank size index)))
  (rank rank)
  (size size)
  (index index))

(defstruct (LoopVariable
	    (:constructor
		make-lvariable (tensor depends-on)))
  (tensor tensor)
  (depends-on depends-on))

(defun maybe> (x y)
  (if (and (numberp x) (numberp y))
      (> x y)
      T))

(defun maybe< (x y)
  (if (and (numberp x) (numberp y))
      (< x y)
      T))

(defun generate-c-kernel (function-name shapes variables abstract-loop instructions
			  &aux
			    (multi-threading-thresholds 128)
			    (iters nil)
			    (indices (iterator-symbols (length abstract-loop)))
			    (indent-count 0)
			    (tiling-p (>= (length abstract-loop) 3)))
  
  (with-compiling-mode
    ;; place-toplevel-form appends these forms:
    ;;  - includes
    ;;  - header
    ;;  - macros
    (place-toplevel-form function-name shapes variables)
    (write-c-line "~a { ~%" (cFunction function-name shapes variables))
    
    (with-indent 4
      ;; [ADD] First touching

      (loop for rank       upfrom 0
	    for index-char in indices
	    for loop       in abstract-loop do
	      (case (aloop-mode loop)
		(:batch
		 (push (make-iteration rank (aloop-size loop) index-char) iters))
		(T
		 (push (make-iteration rank (aloop-element-n loop) index-char) iters))))
      (setq iters (reverse iters))

      (let* ((deps (map 'list #'solve-depends-on variables))
	     (isecs)
	     (not-isecs)
	     (loop-strategy)
	     (vars (map 'list #'make-lvariable variables deps)))

	(flet ((isec-helper (list1 list2)
		 (intersection list1 list2 :test #'equal)))
	  (setq isecs     (reduce #'isec-helper deps))
	  (setq not-isecs (loop for rank upfrom 0 below (dims (car variables))
				unless (find rank isecs)
				  collect rank)))

	(when (not (null not-isecs))
	  (setq tiling-p nil))

	(setq isecs
	      (sort
	       (loop for i in isecs
		     collect (nth i iters))
	       #'maybe>
	       :key #'iteration-size))
	
	(setq not-isecs
	      (sort
	       (loop for i in not-isecs
		     collect (nth i iters))
	       #'maybe>
	       :key #'iteration-size))

	;; not-isecs:
	;;  to maximize the locality of memory, and use of L1/L2 cache
	;;  Sort by strides
	(when tiling-p
	  (flet ((cost (rank)
		   (let ((strides
			   (map 'list
				#'(lambda (v)
				    (let ((out (cStride v rank)))
				      (if (numberp out)
					  out
					  (if (string= "0" out)
					      0
					      most-positive-fixnum))))
				(list (car variables)))))
		     (if (find 1 strides)
			 -1
			 (apply #'* strides)))))
	    (setq isecs
		  (sort
		   isecs
		   #'(lambda
			 (x y)
		       (> (cost (iteration-rank x))
			  (cost (iteration-rank y))))))))
	
	(setq
	 loop-strategy
	 `(,@isecs
	   ,@not-isecs))

	(flet ((step-rank (rank var)
		 (setf (loopvariable-depends-on var)
		       (delete rank (loopvariable-depends-on var))))
	       (determined-p (var)
		 (null (loopvariable-depends-on var))))
	  (loop with placed = (make-hash-table) ;; ID -> PTR_NAME
		with loop-stacks = nil
		for iterator in loop-strategy
		for nth upfrom 0 do
		  (let ((*indent-width* (+ 4 (* 4 indent-count)))
			(delete-loop-p
			  (and (numberp (iteration-size iterator))
			       (= 1 (iteration-size iterator)))))
		    (push iterator loop-stacks)
		    (mapc #'(lambda (v)
			      (step-rank (iteration-rank iterator) v))
			  vars)
		    
		    (when (and
			   (not delete-loop-p)
			   (= indent-count 0)
			   (maybe> (iteration-size iterator) multi-threading-thresholds))
		      (write-c-line "#pragma omp parallel for~%"))

		    (when (and
			   (= indent-count 0)
			   (not delete-loop-p)
			   (numberp (iteration-size iterator))
			   (<= (iteration-size iterator) multi-threading-thresholds))
		      (write-c-line "#pragma omp unroll full~%"))

		    (when (and
			   tiling-p
			   (not delete-loop-p)
			   (= 2 (- (length abstract-loop) nth)))
		      (write-c-line "#pragma omp tile sizes(8, 2)~%")
		      (write-c-line "~%")
		      )

		    (if delete-loop-p
			(write-c-line
			 "uint32_t ~a=0;~%"
			 (iteration-index iterator))
			(progn
			  (incf indent-count)
			  (write-c-line
			   "for (uint32_t ~a=0;~a<~a;~a++) {~%"
			   (iteration-index iterator)
			   (iteration-index iterator)
			   (c-name (format nil "~a" (iteration-size iterator)))
			   (iteration-index iterator))))
		    
		    (dolist (v vars)
		      (when (and
			     (not (= nth (1- (length loop-strategy))))
			     (not
			      (gethash
			       (tensor-id (loopvariable-tensor v))
			       placed))
			     (determined-p v))
			;; [Fix] Private Var?
			(let ((name (format nil "~(~a~)" (gensym))))			
			  (write-c-line
			   "~a ~a = ~a;~%"
			   (dtype->ctype (dtype (loopvariable-tensor v)))
			   name
			   (cAref-with-ranks
			    (loopvariable-tensor v)
			    (map 'list #'iteration-index loop-stacks)
			    (map 'list #'iteration-rank  loop-stacks)))
			  (setf (gethash (tensor-id (loopvariable-tensor v)) placed) name))))

		    ;; Finally, rendering instructings
		    (when (= nth (1- (length loop-strategy)))
		      (let ((*indent-width* (* 4 (1+ indent-count))))
			(dolist (inst instructions)
			  (render-instruction
			   inst
			 indices
			 placed))))))))
      ;; Closing Brackets
      (loop for *indent-width* downfrom (* 4 indent-count) to 0 by 4 do
	(write-c-line "}~%")))))

(defun invoke-compiler (function-name instructions)
  "Compiles to C Kernel.

kernel = instructions[last](... instructions[1](instructions[0](Arguments)))

Inputs:
 - function-name[symbol]
 - instructions[list] a list of instruction

Return:
 JIT-Compiled-Kernel"
  (declare (type list instructions))

  ;; solve-loop-order:
  ;;  Creates an blueprint of optimized loop order
  ;;  This compiler basically follows its instruction, generating corresponding loops in C.
  (let* ((variables (collect-variables instructions))
	 ;; Keep Orders
	 ;; No Loop Collapse;
	 ;; [TODO] set :mode=:polyhedral
	 (abstract-loop (solve-loop-order variables 1 T :mode :runtime))
	 (adjustable-shape))

    (dolist (tensor variables)
      (dolist (shape (shape tensor))
	(when (and (symbolp shape)
		   (not (find shape adjustable-shape)))
	  (push shape adjustable-shape))))

    (let ((source (generate-c-kernel function-name adjustable-shape variables abstract-loop instructions)))
      (setf *lazy-c-source* (format nil "~a~%~a" *lazy-c-source* source))
      (make-jit-compiled-kernel
       :name             function-name
       :args             variables
       :dynamic-symbols  adjustable-shape
       :body             source))))

(defun make-jit-compiled-op (function-name instructions)
  (let* ((jit-kernel (invoke-compiler function-name instructions))
	 (args       (map 'list #'tensor-id (jit-args jit-kernel)))
	 (out        (tensor-id (instruction-displace-to (car (last instructions)))))
	 (inputs     (gensym)))
    (assert (position out args)
	    ()
	    "Assertion Failed: ~a weren't appeared in ~a when jit-compiling ~a" out args instructions)
    (compile
     nil
     `(lambda (&rest ,inputs)
	(let* (,@(loop for nth upfrom 0
		       for arg in args
		       collect
		       `(,arg (nth ,nth ,inputs))))
	  (funcall ,(jit-funcall-form jit-kernel) ,@args)
	  (nth ,(position out args) ,inputs))))))


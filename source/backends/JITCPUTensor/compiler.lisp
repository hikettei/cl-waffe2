
(in-package :cl-waffe2/backends.jit.cpu)

;; [TODO]
;; With enough time, we could introduce these features to a JIT Compiler:
;;  - Polyhedral Compiler
;;  - Tiling, Unrolling
;;  - considering L1/L2 caching, reducing
;; As of this writing, optimization techniques we provide are limited to:
;;  - Memory Locality(L1/L2 Cache)
;;  - Tiling, Unrolling, OpenMP using

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

(defun place-toplevel-form (cffi-call-name shapes tensors dlist)
  "Places headers, function definition and macros."

  ;; Ensures that this call is the first time.
  ;; Since only required once when compiling.
  (when (string= *lazy-c-source* "")
    (write-buff "~%#pragma simd~%")
    ;;(write-buff "#pragma GCC optimize (\"O3\")~%")
    
    (loop for include in *includes* do
      (write-buff "#include <~a>~%" include))

    (when *use-open-mp*
      (write-buff "#include <omp.h>

int get_threads();
int get_threads() { return omp_get_max_threads(); }

")))
  
  (write-buff "~%~a;~%~%" (cFunction cffi-call-name shapes tensors :displace-to-list dlist)))

(defun get-threads () (cffi:foreign-funcall "get_threads" :int))

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
  (declare (ignore make-me-private))
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
  "
<<Headers>>
for(uint32_t index=0;index<size;index++) {
    <Instructions>
        ....
    <Instructions>
    <Iteration[n+1]>
}"
  (rank rank)
  (size size)
  (index index)

  (delete-p     nil)
  (instructions nil :type list)
  (headers      nil :type list))

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

(defun make-iteration-schedule (variables abstract-loop tiling-p indices
				&aux
				  (iters nil))
  "Shuffles the order of abstract-loop to maximize the use of L1/L2 Cache."
  (loop for rank       upfrom 0
	for index-char in indices
	for loop       in abstract-loop do
	  (case (aloop-mode loop)
	    (:batch
	     (push (make-iteration rank (aloop-size loop) index-char) iters))
	    (T
	     (push (make-iteration rank (aloop-element-n loop) index-char) iters))))
  (setq iters (reverse iters))

  ;; Detecting reductions
  (let* ((deps (map 'list #'solve-depends-on variables))
	 (isecs)
	 (not-isecs)
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

    ;; Loop Priority (tmp):
    ;; <<Reductions>>
    ;; <<Larger  strides(e.g.: batch-size)>>
    ;; <<Smaller strides>>
    (values `(,@isecs ,@not-isecs) vars)))

(defparameter *solved-as-zero* nil)
(defun set-loop-blueprint! (loop-schedules
			    vars
			    tiling-p
			    compiling-instructions
			    indices
			    &aux
			      (multi-threading-thresholds 128))
  (flet ((step-rank (rank var)
	   (setf (loopvariable-depends-on var)
		 (delete rank (loopvariable-depends-on var))))
	 (determined-p (var)
	   (null (loopvariable-depends-on var))))
    (loop with *solved-as-zero* = nil
	  with placed = (make-hash-table) ;; ID -> PTR_NAME
	  with loop-stacks = nil
	  for iterator in loop-schedules
	  for nth upfrom 0 do
	    (symbol-macrolet ((headers
				(iteration-headers iterator))
			      (instructions
				(iteration-instructions iterator)))
	      (let ((delete-loop-p
		      (and
		       (numberp (iteration-size iterator))
		       (= 1 (iteration-size iterator)))))
		(push iterator loop-stacks)
		
		(mapc
		 #'(lambda (v)
		     (step-rank (iteration-rank iterator) v))
		 vars)	       
		
		(when (and
		       (not delete-loop-p)
		       (= nth 0)
		       (maybe> (iteration-size iterator) multi-threading-thresholds))
		  (push "#pragma omp parallel for" headers))

		(when (and
		       (= nth 0)
		       (not delete-loop-p)
		       (numberp (iteration-size iterator))
		       (<= (iteration-size iterator) multi-threading-thresholds))
		  (push "#pragma omp unroll partial(4)" headers))

		(when (and
		       tiling-p
		       (= 2 (- (length loop-schedules) nth)))
		  (let ((determined-iters (butlast loop-schedules 2))
			(nd-iters         (last    loop-schedules 2)))
		    (dolist (v vars)
		      (let ((name (format nil "~(~a~)" (gensym "p"))))
			(push
			 (format
			  nil
			  "~a* ~a = &~a;"
			  (dtype->ctype (dtype (loopvariable-tensor v)))
			  name
			  (cAref-with-ranks
			   (loopvariable-tensor v)
			   (map 'list #'iteration-index determined-iters)
			   (map 'list #'iteration-rank  determined-iters)))
			 instructions)
			(setf
			 (gethash (tensor-id (loopvariable-tensor v)) placed)
			 #'(lambda ()
			     (format nil
				     "~a"
				     (cAref-with-ranks
				      (loopvariable-tensor v)
				      (map 'list #'iteration-index nd-iters)
				      (map 'list #'iteration-rank  nd-iters)
				      :name name))))))))

		(when (= nth (1- (length loop-schedules)))
		  (push "#pragma omp unroll partial(4)" headers))

		(when delete-loop-p
		  (setf (iteration-delete-p iterator) t)
		  (push (iteration-index iterator) *solved-as-zero*))
		
		(dolist (v vars)
		  (when (and
			 (not (= nth (1- (length loop-schedules))))
			 (not
			  (gethash
			   (tensor-id (loopvariable-tensor v))
			   placed))
			 (determined-p v))
		    ;; [Fix] Private Var?
		    (let ((name (format nil "~(~a~)" (gensym))))
		      (push
		       (format nil
			       "~a* ~a = &~a;"
			       (dtype->ctype (dtype (loopvariable-tensor v)))
			       name
			       (cAref-with-ranks
				(loopvariable-tensor v)
				(map 'list #'iteration-index loop-stacks)
				(map 'list #'iteration-rank  loop-stacks)))
		       instructions)
		      (setf (gethash (tensor-id (loopvariable-tensor v)) placed)
			    (format nil "~a[0]" name)))))

		(when (= nth (1- (length loop-schedules)))
		  (dolist (inst compiling-instructions)
		    (push
		     (render-instruction
		      inst
		      indices;;(map 'list #'iteration-index loop-schedules)
		      placed)
		     instructions))))))))

;; [FixME] When the generated kernel is parallelized by OpenMP
;; local variables should be declared as a private variable otherwise conflicts
(defun generate-c-kernel (function-name
			  variables
			  abstract-loop
			  instructions
			  &aux			    
			    (indices (iterator-symbols (length abstract-loop)))
			    (tiling-p (>= (length abstract-loop) 3)))
  "Rerturn -> (values code[string] dynamic-shape-envolved[list])"
  (let ((*dynamic-shape-envolved*))
    (multiple-value-bind (loop-schedule lvars)
	(make-iteration-schedule variables abstract-loop tiling-p indices)

      (dolist (iter loop-schedule)
	(let ((size (iteration-size iter)))
	  (maybe-symbol size)))
      
      (set-loop-blueprint! loop-schedule lvars tiling-p instructions indices)
      
      (values       
       (with-compiling-mode
	 ;; place-toplevel-form appends these forms:
	 ;;  - includes
	 ;;  - header
	 ;;  - macros
	 (let ((dlist ;; displace-to-list
		 (loop for inst in instructions
		       collect
		       (tensor-id (instruction-displace-to inst)))))
	   (place-toplevel-form function-name *dynamic-shape-envolved* variables dlist)
	   (write-c-line "~a { ~%" (cFunction
				    function-name
				    *dynamic-shape-envolved*
				    variables
				    :displace-to-list
				    dlist)))

	 (loop for iter in loop-schedule
	       for nth upfrom 0
	       for *indent-width* upfrom 4 by 4 do
		 ;; <Header>
		 ;; Instruction
		 ;; Next Iterations
		 (dolist (header (reverse (iteration-headers iter)))
		   (write-c-line "~a~%" header))

		 (write-c-line
		  "for (uint32_t ~a=0;~a<~a;~a++) {~%"
		  (iteration-index iter)
		  (iteration-index iter)
		  (c-name (format nil "~a" (iteration-size  iter)))
		  (iteration-index iter))
		 
		 (dolist (inst   (reverse (iteration-instructions iter)))
		   (write-c-line "    ~a~%" inst)))
	 
	 ;; Closing Brackets
	 (loop for iter in loop-schedule
	       for *indent-width* downfrom (* 4 (length loop-schedule)) by 4 do
		 (write-c-line "}~%"))
	 (write-c-line "}"))
       *dynamic-shape-envolved*))))

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
	 (abstract-loop (solve-loop-order variables 1 T :mode :runtime)))

    (multiple-value-bind (source adjustable-shape)
	(generate-c-kernel function-name variables abstract-loop instructions)
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


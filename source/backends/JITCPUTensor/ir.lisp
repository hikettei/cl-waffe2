
(in-package :cl-waffe2/backends.jit.cpu)

;; On compiling, we gather tensors which envolved in compiling to *compiled-tensor*.
(defvar *compiled-tensors* nil "An list of variables used in the computation node.")

(defun add-variable (tensor)
  "Adds the given tensor to *compiled-tensors* if there's no duplicate"
  (unless (find (tensor-id tensor) *compiled-tensors* :key #'tensor-id)
    (push tensor *compiled-tensors*)))

;;
;; Trans: opAST -> <opAST>, Scalar, Tensor, nil
;; opAST  = {Variable, Leaves}
;;

(deftype ast-variable-types ()
  `(and keyword (member :opAST :scalar :tensor :null)))

(defstruct (opAST
	    (:constructor make-opAST (operation &rest args)))
  "
    [car args]
        |
an list of AST_Variable
"
  (car  operation :type (or JITCPUScalarTensor JITCPUTensor))
  (args args :type list))

(defstruct (AST-Variable
	    (:constructor make-ast-variable
		(content &aux (type (->op-type content)))))
  "The end of opAST node."
  (type type :type ast-variable-types)
  (content content :type (or null opAST JITCPUScalarTensor JITCPUTensor)))

(defun ->op-type (obj)
  (typecase obj
    (opAST :opAST)
    (JITCPUTensor :tensor)
    (JITCPUScalarTensor  :scalar)
    (null :null)
    (T (error "Detected unknown type of variable: ~a" obj))))

;; Memo:
(defun tensor-using-variables (toplevel) ;; = tensor-variables
  (declare (type (or JITCPUTensor JITCPUScalarTensor) toplevel))
  ;;  [X]  [Y]
  ;;    \  /
  ;;   [MOVE] - toplevel
  (let* ((variables (tensor-variables toplevel))
	 (ignore-first-p (and (movetensor-p (tensor-backward toplevel))
			      (movetensor-ignore-me (tensor-backward toplevel)))))
    (if ignore-first-p
	(cdr variables)
	variables)))

;; [FixME]: Circulation of !copy
;;  sin(x)
;;
;;     a
;;   /   \
;; sin  copy ... <- copy should be detached from nodes.
;;  | <- |
;;  |
;; out
;;

(defun confirm-compiling-area (toplevel)
  "Tracing the previous variables, returns AST of compiling region."
  (declare (type (or JITCPUScalarTensor JITCPUTensor) toplevel))
  ;; register variables
  
  (if (and (movetensor-p         (tensor-backward toplevel))
	   (movetensor-ignore-me (tensor-backward toplevel)))
      ;; MoveTensorNode: X[~] OUT[~] -> X[~]
      ;; If it was pruned by in-place mutation, X is never allocated/used.
      (add-variable (second (tensor-variables toplevel)))
      
      ;; Otherwise, we can collect envolved tensors normally:
      (loop for var in (tensor-variables toplevel)
	    do (add-variable var)))

  ;; Explore JITAble Nodes deeper:
  (apply #'make-opAST toplevel
	 (loop for called-var in (tensor-variables toplevel)
	       if (or (apply-compile-p called-var toplevel)
		      (detach-p called-var))
		 collect (make-ast-variable called-var)
	       else
		 collect (make-ast-variable
			  (confirm-compiling-area called-var)))))

(defun viz-ast (top &key (indent 0))
  (dotimes (i indent) (princ " "))
  (format t "~a~%" (blueprint-opecode (tensor-backward (opAST-car top))))
  (dolist (arg (opAST-args top))
    (when (eql (ast-variable-type arg) :opAST)
      (viz-ast (ast-variable-content arg) :indent (+ 2 indent)))))

;; ~~ TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Fuse operations ... would be realised by composing ops  (e.g.: apply(apply(x)))
;; Delete: A ton of intermidate node, that is,: A=B
;; However, some A=B nodes are needed and shouldn't be pruned, it interrupts optimizing generating codes...
;; Anyway, confirm the very least work and then think about it.
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; [modify] -> [apply] could be reduced?

(defun op->inst (opAST)
  (apply #'translate-op (blueprint-opecode (tensor-backward (opAST-car opAST))) opAST
	 (loop for var in (opAST-args opAST)
	       if (eql (ast-variable-type var) :opAST)
		 collect (opAST-car (ast-variable-content var))
	       if (or (eql (ast-variable-type var) :tensor)
		      (eql (ast-variable-type var) :scalar))
		 collect (ast-variable-content var))))

(defun ir->c (opAST)
  "Recursively this function explores opAST, generating and writing C code to buffer."
  (declare (type opAST opAST))

  (when (null (tensor-backward (opAST-car opAST)))
    (return-from ir->C))
  
  
  (loop for var in (opAST-args opAST)
	if (eql (ast-variable-type var) :opAST)
	  do (ir->C (ast-variable-content var)))
  
  (let* ((form (op->inst opAST))
	 ;; Identify the usage of "=" in the computation node
	 (save-for-backward-p ;; "=" is intended to make save4bw?
	   (and (equal "=" (instruction-fname form))
		(system-lazy-read-save-for-backward (opAST-car opAST))))
	 (copy-for-safety ;; "=" is intended to avoid side effects on ExistTensor?
	   (and (equal "=" (instruction-fname form))
		;; (axpy! 1.0 ExistTensor ExistTensor) isn't allowed
		;; instead, (axpy! 1.0 ExistTensor (copy ExistTensor))
		(eql (tensor-attribute (car (instruction-args form))) :input))))
    

    (write-c-line "~%")
    (case (Instruction-type form)
      (:modify
       ;; A[...] += A[...]; // comments if any
       
       (write-c-line "// [modify] A ~a B~%"
		     (instruction-fname form))
       (write-c-line "~a ~a ~a;~a~%"
		     (cAref (instruction-displace-to form) :pointer t)
		     (instruction-fname form)
		     (cAref (car (instruction-args form)) :pointer t)
		     (cond
		       ;; The operation "=" is placed for saving for backward
		       (save-for-backward-p " // saving for backward")
		       ;; The operation "=" is placed for protecting tensors
		       ;; (i.e.: ExistTensor isn't allowed to be in-place)
		       (copy-for-safety     " // in-place guard for :exist tensors")
		       ((equal (instruction-fname form) "=") " // intended copy")
		       (T ""))))
      (:apply
       ;; A[...] = f(A[...], B[...]);
       (write-c-line "// [apply]  ~a~%"
		     (instruction-fname form))
       (write-c-line "~a = ~a(~a);~%"
		     (cAref (instruction-displace-to form) :pointer t)
		     (instruction-fname form)
		     (with-output-to-string (out)
		       (loop for arg in (instruction-args form)
			     for i upfrom 0
			     do (princ (cAref arg :pointer t) out)
			     unless (= i (1- (length (instruction-args form))))
			       do (princ  ", " out)))))
      
      (:set
       ;;         type* variable = value
       ;; moves variable -> value with no copies.
       (if (equal "=" (instruction-fname form))
	   (progn
	     ;;
	     ;; float* XXX1 = XXX2;
	     ;;

	     ;; register in-place mutation at toplevel, so compiler can synchronize them later without copying.
	     (register-in-place-mutation
	      ;; Sync: opAST-car <- car args
	      (opAST-car opAST) (car (instruction-args form)))
	     
	     (write-c-line "// [set]    in-place mutation~%")
	     (write-c-line "int32_t ~a_STRIDE = ~a_STRIDE;~%"
			   (tensor-id (opAST-car opAST))
			   (tensor-id (car (instruction-args form)))))
	   (write-c-line "// [set]    A = B~%"))
       
       (write-c-line "~a* ~a ~a ~a;~%"
		     (dtype->ctype (dtype (opAST-car opAST)))
	 	     (tensor-id (opAST-car opAST))
	 	     "="
		     (tensor-id (car (instruction-args form)))))
      (:ignore
       
       ))))


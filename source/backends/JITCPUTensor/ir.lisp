
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

(defun confirm-compiling-area (toplevel)
  "Tracing the previous variables, returns AST of compiling region."
  (declare (type (or JITCPUScalarTensor JITCPUTensor) toplevel))

  (if (and (movetensor-p (tensor-backward toplevel))
	   (movetensor-ignore-me (tensor-backward toplevel)))
      ;; MoveTensorNode: X[~] OUT[~] -> X[~]
      ;; If pruned, X is never allocated/used.
      (add-variable (second (tensor-variables toplevel)))
      
      ;; Otherwise, we can collect envolved tensors normally:
      (loop for var in (tensor-variables toplevel)
	    ;; if (apply-compile-p toplevel var)
	    do (add-variable var)))

  ;; Explore JITAble Nodes deeper:
  (apply #'make-opAST toplevel
	 (loop for called-var in (tensor-variables toplevel)
	       if (apply-compile-p toplevel called-var)
		 collect (make-ast-variable called-var)
	       else
		 collect (make-ast-variable
			  (confirm-compiling-area called-var)))))

;; ~~ TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Fuse operations ... would be realised by composing ops  (e.g.: apply(apply(x)))
;; Delete: A ton of intermidate node, that is,: A=B
;; However, some A=B nodes are needed and shouldn't be pruned, it interrupts optimizing generating codes...
;; Anyway, confirm the very least work and then think about it.
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun ir->c (opAST)
  "Recursively this function explores opAST, generating and writing C code to buffer."
  (declare (type opAST opAST))

  (let ((code (blueprint-opecode (tensor-backward (opAST-car opAST)))))
    (loop for var in (opAST-args opAST)
	  if (eql (ast-variable-type var) :opAST)
	    do (ir->C (ast-variable-content var)))

    
    (let ((form
	    (apply #'translate-op code opAST
		   (loop for var in (opAST-args opAST)
			 if (eql (ast-variable-type var) :opAST)
			   collect (opAST-car (ast-variable-content var))
			 if (or (eql (ast-variable-type var) :tensor)
				(eql (ast-variable-type var) :scalar))
			   collect (ast-variable-content var)))))

      (case (Instruction-type form)
	(:modify
	 ;; A[...] += A[...];
	 (write-c-line "~a ~a ~a;~%"
		       (cAref (instruction-displace-to form) :pointer t)
		       (instruction-fname form)
		       (cAref (car (instruction-args form)) :pointer t))
	 (write-c-line "~a ~a ~a;~%"
		       (cAref (opAST-car opAST) :pointer t)
		       "="
		       (cAref (instruction-displace-to form) :pointer t))
	 )
	(:apply
	 ;; A[...] = f(A[...], B[...]);
	 (write-c-line "~a = ~a(~a);~%"
		       (cAref (instruction-displace-to form) :pointer t)
		       (instruction-fname form)
		       (with-output-to-string (out)
			 (loop for arg in (instruction-args form)
			       for i upfrom 0
			       do (princ (cAref arg :pointer t) out)
			       unless (= i (1- (length (instruction-args form))))
				 do (princ  ", " out))))

	 
	 (write-c-line "~a ~a ~a;~%"
		       (cAref (opAST-car opAST) :pointer t)
		       "="
		       (cAref (instruction-displace-to form) :pointer t))
	 )
	
	(:set

	 (error ":set isn't available currently")
	 
	 ;; (write-c-line "~a ~a ~a;~%"
	 ;;	       (cAref (opAST-car opAST))
	 ;;	       "="
	 ;;	       (cAref (instruction-displace-to form)))
	 )

	(:ignore
	 
	 )))))


(in-package :cl-waffe2/backends.jit.cpu)


(defvar *compiled-tensors* nil "An list of variables used in the computation node.")

(defun add-variable (tensor)
  ;; If the backward is MoveTensorNode
  ;; ignore this function on some conditions
  (unless (find tensor *compiled-tensors*)
    (push tensor *compiled-tensors*)))

(deftype ast-variable-types () `(and keyword (member :opAST :scalar :tensor :null)))

(defstruct (iSeq
	    (:constructor make-iseq (code displace-out-to)))
  (code code :type (or symbol list))
  (displace-out-to displace-out-to :type (or symbol list)))

(defstruct (opAST
	    (:constructor make-opAST (operation &rest args)))
  "opAST is a data structure which is:
[car args]
      |
    an list of AST_Variable"
  (car  operation :type (or JITScalarTensor JITCPUTensor))
  (args args :type list))

(defstruct (AST-Variable
	    (:constructor make-ast-variable
		(content &aux (type (->op-type content)))))
  (type type :type ast-variable-types)
  (content content :type (or null opAST JITScalarTensor JITCPUTensor)))

(defun ->op-type (obj)
  (typecase obj
    (opAST :opAST)
    (JITCPUTensor :tensor)
    (JITScalarTensor  :scalar)
    (null :null)
    (T (error "Detected unknown type of variable: ~a" obj))))

(defun confirm-compiling-area (toplevel)
  "Tracing the previous variables, returns AST of compiling region."
  (declare (type (or JITScalarTensor JITCPUTensor) toplevel))

  (add-variable toplevel)
  (let* ((variables (tensor-variables toplevel)))
    (apply #'make-opAST toplevel
	   (loop for called-var in variables
		 if (apply-compile-p toplevel called-var)
		   collect (progn
			     (add-variable called-var)
			     (make-ast-variable called-var))
		 else
		   collect (make-ast-variable
			    (confirm-compiling-area called-var))))))

(defun ir->c (opAST)
  (declare (type opAST opAST))

  ""
  )

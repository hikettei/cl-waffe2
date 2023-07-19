
(in-package :cl-waffe2/backends.jit.lisp)

(defvar *compiled-tensors* nil "Tensor1 Tensor2...")
(defvar *compiled-tensors-aref* nil "(aref Tesnor1 i) (aref Tensor2 i) ...")
(defvar *subscript-char* nil)
;; (defmethod (eql ...) or pattern match)

;; TODO: Eliminate unused MoveTensorNode and allocation with it.
;; AVX2 Isn't wokirng? Supper slow -> No, compute-stepby is included to complied code.
;; Except the usage of no-grad
;; setf使わない代わりに高速に動作とか？
;; setfのアクセスをキャッシュするとかしないと速くならない

(defun ->op-type (obj)
  (typecase obj
    (opAST :opAST)
    (JITLispTensor :tensor)
    (ScalarTensor  :scalar)
    (null :null)
    (T (error "Detected unknown type of variable: ~a" obj))))

(deftype ast-variable-types () `(and keyword (member :opAST :scalar :tensor :null)))
    
(defstruct (opAST
	    (:constructor make-opAST (operation &rest args)))
  "opAST is a data structure which is:
[car args]
      |
    an list of AST_Variable"
  (car  operation :type (or ScalarTensor JITLispTensor))
  (args args :type list))

(defstruct (AST-Variable
	    (:constructor make-ast-variable
		(content &aux (type (->op-type content)))))
  (type type :type ast-variable-types)
  (content content :type (or null opAST ScalarTensor JITLispTensor)))

(defun add-variable (tensor)
  ;; TODO: Ignore MoveTensorNode
  (unless (find tensor *compiled-tensors*)
    ;; ScalarTensor isn't iteratible
    (when (not (typep tensor 'ScalarTensor))
      (push tensor *compiled-tensors*))))

(defun confirm-compiling-area (toplevel)
  "Tracing the previous variables, returns AST of compiling region."

  (declare (type JITLispTensor toplevel))

  (let* ((variables (tensor-variables toplevel)))
    (apply #'make-opAST toplevel
	   (loop for called-var in variables
		 do (add-variable called-var)
		 if (apply-compile-p toplevel called-var)
		   collect (make-ast-variable called-var)
		 else
		   collect (make-ast-variable
			    (confirm-compiling-area called-var))))))

(defun invoke-compiler! (current-node variable next-var)
  (declare (type LispJIT-Blueprint current-node)
	   (type JITLispTensor variable)
	   (type (or null JITLispTensor) next-var))
  (declare (ignore current-node next-var))

  (let* ((*compiled-tensors* `(,variable))
	 (*subscript-char* (gensym "Index"))
	 (compiling-area (confirm-compiling-area variable)))

    ;; The form below is expanded into:
    ;;
    ;; (let ((vec1 (tensor-vec a))
    ;;       (vec2 (tensor-vec b))
    ;;              ...
    ;;
    ;;   (loop-with-view ...
    ;;       (setf (aref vec1 index) (sin (aref ...)))))
    ;;

    `(progn
       ,(with-allocating-vectors *compiled-tensors*
	  (with-expand-call-with-view *compiled-tensors* *subscript-char*
	    (trace-and-compile! compiling-area)))
       
       (setf (cl-waffe2/vm.generic-tensor::statecontainer-forward-result
	      (tensor-state ,(tensor-id variable)))
	     (list ,(tensor-id variable))))))


(defun tensor-vec-id (tensor)
  (declare (type AbstractTensor tensor))
  (symb (tensor-id tensor) '-vec))

(defun tensor-stride-id (tensor)
  (symb (tensor-id tensor) '-stride))

(defun tensor-offset-id (tensor)
  (symb (tensor-id tensor) '-offset))

(defun with-allocating-vectors (tensors body)
  `(let (,@(loop for tensor in tensors
		 collect `(,(tensor-vec-id tensor) (tensor-vec ,tensor))))
     (declare ,@(loop for tensor in tensors
		      collect `(type (simple-array ,(dtype->lisp-type (dtype tensor)) (*)) ,(tensor-vec-id tensor)))
	      (ignorable ,@(map 'list #'tensor-vec-id tensors)))
     ,body))

;; TODO: Do iteration with each strides/offsets
(defmacro with-expand-call-with-view (tensors index-char &body body)
  (let ((views-area (gensym)))
    `(call-with-view
      #'(lambda (&rest ,views-area)
	  (let ((*compiled-tensors-aref* (view->accessors ,views-area ,index-char)))
	    `(loop for ,,index-char fixnum upfrom 0 below ,(size-of (car ,views-area) 0)
		   do (let (,@(loop for view in ,views-area
				    for tensor in ,tensors
				    collect `(,(tensor-stride-id tensor) (the fixnum ,(stride-of view 0))))
			    ,@(loop for view in ,views-area
				    for tensor in ,tensors
				    collect `(,(tensor-offset-id tensor) (the fixnum ,(offset-of view 0)))))
			,,@body))))
      ,tensors
      :at-least-dim 1)))

(defun view->accessors (views index-symbol)
  "View -> '(aref tensor i)"
  (let ((result (make-hash-table)))
    (loop for view   in views
	  for tensor in *compiled-tensors*
	  do (let* ((key (tensor-id tensor))
		    (reader `(aref ,(tensor-vec-id tensor)
				   ;; adding offsets
				   ;; multiplying strides
				   (+ ,(tensor-offset-id tensor)
				      (the fixnum
					   (* ,(tensor-stride-id tensor)
					      ,index-symbol))))))
	       (setf (gethash key result) reader)))
    result))

(defun expand-aref (tensor)
  (declare (type AbstractTensor tensor))
  (gethash (tensor-id tensor) *compiled-tensors-aref*))

(defgeneric implement-op (opcode opAST &rest arguments))

(defvar *instructions* nil)

(defun trace-and-compile! (compile-toplevel)
  (declare (type opAST compile-toplevel))
  (let ((*instructions* nil))
    (explore-and-compile! compile-toplevel)
    `(progn ,@(reverse *instructions*))))

(defun explore-and-compile! (compile-toplevel)
  (declare (type opAST compile-toplevel))

  ;; If the end of node?

  (when (null (tensor-backward (opAST-car compile-toplevel)))
    (return-from explore-and-compile!
      (expand-aref (opAST-car compile-toplevel))))
  (let* ((code (blueprint-opecode (tensor-backward (opAST-car compile-toplevel)))))

    (loop for var in (reverse (opAST-args compile-toplevel))
	  if (eql (ast-variable-type var) :opAST)
	    collect (explore-and-compile! (ast-variable-content var)))
    
    (push
     `(setf
       ,(expand-aref (opAST-car compile-toplevel))
       ,(apply #'implement-op code compile-toplevel
		 (loop for var in (opAST-args compile-toplevel)
		       if (eql (ast-variable-type var) :opAST)
			 collect (expand-aref (opast-car (ast-variable-content var)))
		       if (eql (ast-variable-type var) :tensor)
			 collect (expand-aref (ast-variable-content var))

		       if (eql (ast-variable-type var) :scalar)
			 collect `(the ,(dtype->lisp-type (dtype (ast-variable-content var))) (tensor-vec ,(ast-variable-content var))))))
	  *instructions*)
    nil))



(in-package :cl-waffe2/backends.jit.lisp)

;; (defmethod (eql ...) or pattern match)

;; TODO: Eliminate unused MoveTensorNode and allocation with it.
;; AVX2 Isn't wokirng? Supper slow -> No, compute-stepby is included to complied code.
;; with-no-grad is MUST, since all copies are ignored

;; Things remained to be optimized:
;; TODO:
;; Lparallel
;; Aggregation of +-*/: (+ (+ 1 1) 1) -> (+ 1 1 1)
;; 同じViewのStride演算を結合する
;; Strideの計算 実行時にViewとShapeが同じだと判明しているものはそのまま使う
;; !sum動く？
;; !sum !mean ugokanai.... ;;
;; ArefをLetでCacheする
;; コンパイルされたコードは大量に最適化の余地があるんだけど、単純にするためにそのままにしておく。
;; The goal: Softmax関数を動かす Adamをこれで実装する
;; Partial compile-test

(defvar *compiled-tensors* nil "Tensor1 Tensor2...")
(defvar *compiled-tensors-aref* nil "(aref Tesnor1 i) (aref Tensor2 i) ...")
(defvar *subscript-char* nil)


(defun ->op-type (obj)
  (typecase obj
    (opAST :opAST)
    (JITLispTensor :tensor)
    (ScalarTensor  :scalar)
    (null :null)
    (T (error "Detected unknown type of variable: ~a" obj))))

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
  (car  operation :type (or ScalarTensor JITLispTensor))
  (args args :type list))

(defstruct (AST-Variable
	    (:constructor make-ast-variable
		(content &aux (type (->op-type content)))))
  (type type :type ast-variable-types)
  (content content :type (or null opAST ScalarTensor JITLispTensor)))

(defun add-variable (tensor toplevel)
  (unless (find tensor *compiled-tensors*)
    ;; ScalarTensor isn't iteratible
    (when (and
	   (not (typep tensor 'ScalarTensor))
	   (find tensor (blueprint-use-var (tensor-backward toplevel))))
      (push tensor *compiled-tensors*))))

;; !Mean isn't working???
(defun with-allocating-vectors (tensors body)
  `(let (,@(loop for tensor in tensors
		 collect `(,(tensor-vec-id tensor) (tensor-vec 
						    ;; If tensor is produced by the result of any nodes?
						    ,(if (typep tensor 'JITLispTensor)
							 ;; If tensor is produced by other devices, read id. otherwise use itself.
							 tensor
							 `(tensor-id ,tensor))))))
     (declare ,@(loop for tensor in tensors
		      collect `(type (simple-array ,(dtype->lisp-type (dtype tensor)) (*)) ,(tensor-vec-id tensor)))
	      (ignorable ,@(loop for tensor in tensors
				 collect (tensor-vec-id tensor))))
     ,body))

;; TODO: Do iteration with each strides/offsets
;; TODO: lparallel
(defmacro with-expand-call-with-view (tensors index-char &body body)
  (let ((views-area (gensym)))
    `(call-with-view
      #'(lambda (&rest ,views-area)
	  (let ((*compiled-tensors-aref* (view->accessors ,views-area ,index-char)))
	    `(let (,@(loop for view in ,views-area
			   for tensor in ,tensors
			   collect `(,(tensor-stride-id tensor) (the fixnum ,(stride-of view 0))))
		   ,@(loop for view in ,views-area
			   for tensor in ,tensors
			   collect `(,(tensor-offset-id tensor) (the fixnum ,(offset-of view 0)))))
	       (declare (ignorable ,@(loop for tensor in ,tensors collect (tensor-stride-id tensor))
				   ,@(loop for tensor in ,tensors collect (tensor-offset-id tensor))))
	       ;; TODO: For larger scal operations, lparallel would be nice
	       ;; (cl-waffe2/threads:parallel-dotimes (p *work-num*)  ...)
	       (dotimes (,,index-char (the fixnum ,(size-of (car ,views-area) 0)))
		 (let (,@(loop for tensor in ,tensors
			       collect `(,(tensor-aref-id tensor) ,(expand-aref tensor))))
		   (declare
		    (type unsigned-byte ,,index-char)
		    (ignorable ,@(loop for tensor in ,tensors collect (tensor-aref-id tensor))))
		   ;; Call aref in advance:
		   ,,@body)))))
      ,tensors
      :at-least-dim 1)))

(defun confirm-compiling-area (toplevel)
  "Tracing the previous variables, returns AST of compiling region."

  (declare (type (or ScalarTensor JITLispTensor) toplevel))

  (let* ((variables (tensor-variables toplevel)))
    (apply #'make-opAST toplevel
	   (loop for called-var in variables
		 do (add-variable called-var toplevel)
		 if (apply-compile-p toplevel called-var)
		   collect (make-ast-variable called-var)
		 else
		   collect (make-ast-variable
			    (confirm-compiling-area called-var))))))

(defun invoke-compiler! (current-node variable next-var)
  (declare (type LispJIT-Blueprint current-node)
	   (type (or ScalarTensor JITLispTensor) variable)
	   (type (or null JITLispTensor) next-var))
  (declare (ignore current-node next-var))

  (let* ((*compiled-tensors* (if (typep variable 'JITLispTensor)
				 `(,variable)
				 nil))
	 (*compiled-tensors-aref* (make-hash-table))
	 (*subscript-char*        (gensym "Index"))
	 (compiling-area          (confirm-compiling-area variable)))

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

(defun tensor-aref-id (tensor)
  (symb (tensor-id tensor) '-aref))

(defun ->force-aref (tensor)
  (expand-aref tensor))

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

(defun trace-and-compile! (compile-toplevel)
  (declare (type opAST compile-toplevel))
  (let* (;;(*tensors-use* nil)
	 (tree (explore-and-compile! compile-toplevel)))
    (print `(setf ,(expand-aref (opAST-car compile-toplevel)) ,tree))))

(defun explore-and-compile! (compile-toplevel)
  (declare (type opAST compile-toplevel))
  ;; If the end of node?

  (when (null (tensor-backward (opAST-car compile-toplevel)))
    (return-from explore-and-compile!
      (tensor-aref-id (opAST-car compile-toplevel))))
  
  (let* ((code (blueprint-opecode (tensor-backward (opAST-car compile-toplevel)))))
    
    (let ((iseq (apply #'implement-op code compile-toplevel
		       (loop for var in (opAST-args compile-toplevel)
			     if (eql (ast-variable-type var) :opAST)
			       collect (explore-and-compile! (ast-variable-content var))
			     
			     if (eql (ast-variable-type var) :tensor)
			       collect (tensor-aref-id (ast-variable-content var))

			     if (eql (ast-variable-type var) :scalar)
			       ;; Cache: accessing to scalar
			       collect (let ((scal-val (ast-variable-content var)))
					 `(the ,(dtype->lisp-type (dtype (ast-variable-content var)))
					       (tensor-vec (read-result ,scal-val))))))))
      (typecase iseq
	(list iseq)
	(iseq (iseq-code iseq))))))




(in-package :cl-waffe2/vm.nodes)


;; TODO: with-embedding-lisp as a alias of this macro.

(defnode (InstantKernelNode (myself call-form)
	  :slots ((call-form :initarg :call-form :type function :reader instant-call-form))
	  :where (A[~] -> A[~])
	  :documentation ""))

(define-impl (InstantKernelNode :device t)
	     :forward ((self x)
		       (let ((result (funcall (instant-call-form self))))
			 (typecase result
			   (list result)
			   (T `(,x)))))
	     :backward ((self dout dx)
			(declare (ignore dx))
			(values dout)))

(defmacro with-instant-kernel (tensor &body body)
  "Creates an instant-kernel following tensor.

This macro is used to embed condition-free Lisp code either in the process of creating a node or after it has been compiled.

Use case:

1. Embedding Lisp Code for building-time.

(setq a (randn `(10 10)))
(with-instant-kernel a
    (print a)) ;; -> (print a) is evaluated

2. Embedding Lisp Code for compile-time.

(setq a (randn `(10 10)))
(with-instant-kernel a
    `(print ,a)) ;; -> (print a) isn't evaluated

(funcall (build *)) ;; -> (print a) will be evaluated.

Note that (equal (with-instant-kernel a) a) is NIL, that is, the returned value of this macro must be followed by a calculation node.

If the return value of Body can be expanded as a macro, the values are compiled together at JIT compile time. Otherwise, the given tensor is returned as is.

TODO: More simple description.
"
  (let ((kernel-name (gensym "InstantKernel")))
    `(flet ((,kernel-name ()
	      ,@body))
       (forward (InstantKernelNode #',kernel-name) ,tensor))))

(defun list-of-keywords-p (list)
  (and (listp list)
       (every #'keywordp list)))

(defmacro supported-dtypes-are (nth &rest dtypes)
  "The macro supported-dtype-are returns a predictor which returns nil if the nth argument's keyword is in dtypes, otherwise t.

This macro is useful when combined with define-impl/reject-p.

For example, when your implementation only supports for :float :double, and wants to restrict the use of the node for other dtypes, this macro can be used like:

(defnode (AnyNode (myself dtype))
         ...)

(define-impl (AnyNode :device CPUTensor
                      :reject-p (supported-dtype-are 0 :float :double))

    ...)

(AnyNode :float)
(AnyNode :uint8)

This means: the first argument of :forward was the dtype of :float or :double, use AnyNode, otherwise, AnyNode is no longer available (use other nodes instead)."
  (declare (type (satisfies list-of-keywords-p) dtypes))
  `#'(lambda (&rest inputs)
       (let ((dtype (nth ,nth inputs)))
	 (not (find dtype ',dtypes)))))


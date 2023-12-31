
(in-package :cl-waffe2/vm.nodes)


;; TODO: with-embedding-lisp as a alias of this macro.

(defnode (InstantKernelNode (myself call-form)
	  :slots ((call-form :initarg :call-form :type function :reader instant-call-form))
	  :where (A[~] -> A[~])
	  :documentation ""))

(defun gensym-id (&rest args) (declare (ignore args)) `(,(gensym)))
(define-impl (InstantKernelNode :device t :cache-id #'gensym-id)
	     :forward ((self x)
		       (let ((result (funcall (instant-call-form self))))
			 (setf (out-scalar-p self) (cl-waffe2/vm.generic-tensor:scalar-p x))
			 (typecase result
			   (list result)
			   (T `(,x)))))
	     :backward ((self dout dx)
			(declare (ignore dx))
			(values dout)))

(defun map-tree (fn tree)
  (let ((tree (funcall fn tree)))
    (if (listp tree)
        (mapcar (lambda (subtree)
                  (map-tree fn subtree))
                tree)
        tree)))

(defmacro with-instant-kernel (tensor &body body)
  "
```lisp
(with-instant-kernel tensor &body body)
```

Continues the computation node following tensor with embedding an `instant-kernel`. `Instant` is Lisp code that can be embedded in compiled functions.

### Embedding Lisp Code for building-time.

```lisp
(setq a (randn `(10 10)))
(with-instant-kernel a
    (print a)) ;; -> (print a) is evaluated
```

### Embedding Lisp Code for compile-time.

```lisp
(setq a (randn `(10 10)))
(with-instant-kernel a
    `(print ,a)) ;; -> (print a) isn't evaluated

(funcall (build *)) ;; -> (print a) will be evaluated.
```

Note that `(equal (with-instant-kernel a) a)` is `NIL`, that is, the returned value of this macro must be followed by a calculation node.

If the return value of `body` can be expanded as a macro, the values are compiled together at JIT compile time. Otherwise, the given tensor is returned as is.
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

(defun invalid-t-usage-p (node inputs)
  (and (some #'cl-waffe2/base-impl::transposed-p inputs)
       (not (subtypep (class-of node) 'cl-waffe2/base-impl:MatmulNode))))

(defun !maybe-move (place tensor &key (deterministic-p nil))
  "Moves the result of backwards, into variables where it was (if deterministic).

Return:
    (values next-tensor moved-p)
    moved-p ... If p, move me to where it was."
  ;; If deterministic-p = t, do in-place.
  (with-no-grad
    (when tensor
      (if (movetensor-p (tensor-backward tensor))
	  (values tensor nil);; the tensor is alread moved into somewhere?
	  (with-shape-checkpoint (:moving nil)
	    (let ((place (if deterministic-p
			     place ;; place=tensor.variables[n]
			     (make-input (shape place) nil
					 :create-from place
					 :scalar-p (scalar-p place)
					 :dtype (dtype place)
					 :order (order place)))))
	      ;; Forcibly moving them.
	      (values (cl-waffe2/base-impl:!move place tensor :force t) t)))))))

(defun detach (tensor &optional (state t))
  (setf (cl-waffe2/vm.generic-tensor::detach-p tensor) state)
  tensor)

(defun nth-subscript (nth)
  "Returns nth alphabet"
  (intern (format nil "Input-~a" (code-char (+ 65 (mod nth 26))))))

(defun ->keyword (symbol)
  (intern (format nil "~a" symbol) "KEYWORD"))

(defun dim->input-shape (dim)
  "3 -> (a b c) 2 -> (a b)"

  (when (>= dim 27)
    (error "Assertion Failed: dim < 27, butgot: ~a" dim))
  
  (loop for i upfrom 0 below dim
	collect (nth-subscript i)))

(defun include~p (composite)
  "Returns t if composite's input definiton has ~"
  (some #'(lambda (x) (some #'(lambda (x) (symbol-eq '~ x)) x)) (composite-input-size composite)))
    


;; [FixME] shouldn't include shape keyword...
;; Fix in the future refactor of :cl-waffe2/vm
(defun tensor-keyname (tensor)
  (symb ;; {BackendName[Dtype]<Permute-Order>}
   '{
   (class-name (class-of tensor))
   '[
   (intern (symbol-name (dtype tensor)))
   ']
   '<
   (intern (format nil "~a~a" (actual-shape tensor) (cl-waffe2/vm.generic-tensor::tensor-permute-order tensor)))
   '>
   '}))


(defun input-det-n-list (composite)
  "returns the number of subscripts ignored ~"
  (loop for i in (composite-input-size composite)
	collect (- (length i) (count '~ i :test #'symbol-eq))))

(defun where->shapes (where)
  "where -> (values ((i j) (j k)) ((i j k)))"
  (multiple-value-bind (x y first-state out-state z) (parse-subscript where)
    (declare (ignore x y z))
    (values first-state out-state)))

(defun collect-initarg-slots (slots constructor-arguments)
  (map 'list #'(lambda (slots)
		 ;; Auto-Generated Constructor is Enabled Only When:
		 ;; slot has :initarg
		 ;; slot-name corresponds with any of constructor-arguments
		 (when
		     (and
		      (find (first slots) (flatten constructor-arguments))
		      (find :initarg slots))
		   slots))
       slots))

(defmacro with-detach-tensors ((tensors) &body body)
  `(let ((detach-state (map 'list #'cl-waffe2/vm.generic-tensor:detach-p ,tensors)))
     (map 'list #'(lambda (x) (setf (cl-waffe2/vm.generic-tensor:detach-p x) t)) ,tensors)
     (unwind-protect
	  (progn ,@body)
       (loop for s in detach-state
	     for tens in ,tensors do
	       (setf (cl-waffe2/vm.generic-tensor:detach-p tens) s)))))

(defun make-grad-gensym () (intern (symbol-name (gensym "Chain")) "KEYWORD"))


(defun parse-broadcasted-shape (shapes)
  (flet ((apply-refine (list)
	   (loop for s in list
		 unless (and *enable-broadcasting-auto* ;; not created by broadcast
			     (equal s -1))
		   collect (if (equal s -1)
			       1
			       s))))
    (map 'list #'apply-refine shapes)))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defnode (System-Lazy-Cons (self a b)
	  :extends (cl-waffe2/base-impl:Rebundant-Node)
	  :where (A[a-size] B[b-size] -> A[a-size] B[a-size] where a-size = (shape a) b-size = (shape b)))
  (setf (ignore-shape-error self) t))

(define-impl-op (System-Lazy-Cons :device t) :forward ((self a b) (values a b)))

(defun !system-lazy-cons (a b)
  (call (System-Lazy-Cons a b) a b))

(defun !system-lazy-values (&rest args)
  (reduce #'!system-lazy-cons args))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun range (start end) (loop for i upfrom start below end collect i))


(defmacro define-and-impl-node ((abstract-name
				 (self &rest constructor-arguments)
				 &key
				   (extends nil)
				   (device t)
				   (cache-when-compiled t)
				   (reject-p nil)
				   (where t)
				   (out-scalar-p nil)
				   (slots nil)
				   (save-for-backward nil)
				   (forward nil)
				   (backward nil)
				   (documentation ""))
				&body constructor-body)
  "
```lisp
(define-and-impl-node (abstract-name
				 (self &rest constructor-arguments)
				 &key
                                   (extends nil)
				   (device t)
				   (cache-when-compiled t)
				   (reject-p nil)
				   (where t)
				   (out-scalar-p nil)
				   (slots nil)
				   (save-for-backward nil)
				   (forward nil)
				   (backward nil)
				   (documentation \"\")))
```

Expands `defnode` and `define-impl` at the same time.
"
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (defnode (,abstract-name (,self ,@constructor-arguments)
	       :extends ,extends
	       :where ,where
	       :out-scalar-p ,out-scalar-p
	       :slots ,slots
	       :save-for-backward ,save-for-backward
	       :backward ,backward
	       :documentation ,documentation)
       ,@constructor-body)
     (define-impl (,abstract-name :device ,device :cache-when-compiled ,cache-when-compiled :reject-p ,reject-p)
		  :save-for-backward ,save-for-backward
		  :forward ,forward)))


(defun element-wise-p (node variables)
  "Return T if the node is bidijective: A[~] B[~] -> A[~]"
  (declare (type AbstractNode node))
  (multiple-value-bind (in-names out-names in-subs out-subs lets)
      (parse-subscript (read-where node))
    (declare (ignore lets))
    (flet ((helper (x)
	     (and (= (length x) 1)
		  (symbol-eq (car x) '~))))
      (and
       (= (length in-names)  2)
       (= (length out-names) 1)
       (every #'helper in-subs)
       (every #'helper out-subs)
       (let ((out-pos (position (car out-names) in-names :test #'symbol-eq)))
	 (and
	  out-pos
	  (values
	   t
	   (case out-pos
	     (0 (nth 1 variables))
	     (1 (nth 0 variables)))
	   (nth out-pos variables))))))))

(defun get-invocations-from-node (node variables)
  (declare (type AbstractNode node))
  (multiple-value-bind (elwisep source-tensors target-tensors) (element-wise-p node variables)
    (when elwisep
      (wf/iter:trace-invocation
       (class-name (class-of node))
       source-tensors
       target-tensors
       :kernel-rank 1
       :collapse t))))


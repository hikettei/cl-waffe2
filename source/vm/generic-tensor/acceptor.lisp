
(in-package :cl-waffe2/vm.generic-tensor)

;; CFFI-Styleの No Overhead generic-function
;; Building Forward/Backward codes

;;(defstruct (StateContainer)) result (計算結果), state, (実行されてない 実行された), form (S式)を格納できる

(defstruct (StateContainer)
  (state :initialized :type (member :initialized :forwarded :backwarded))
  (forward-out-form nil :type list)
  (forward-result   nil :type list)
  (backward-out-form nil :type list)
  (backward-result   nil :type list)

  (forward-n-out  0 :type fixnum)
  (backward-n-out 0 :type fixnum))

(defparameter *node-variables-tmp* nil)
(defparameter *node-parameters-tmp* nil)

(defun construct-variables-table (variables-list)
  "Returns a plist where key and value is tensor's name and tensor's pointer."
  (declare (type list variables-list))
  (let ((out-table `()))
    (mapc
     #'(lambda (tensor)
	 (setf (getf out-table (tensor-name tensor)) tensor))
     variables-list)
    out-table))

(defun construct-forward (toplevel &key (macroexpand nil))
  "TODO: Docstring

Return:
    (values function-body[compiled-function]
            variables[plist]
            parameters[list])"
  (declare (type AbstractTensor toplevel))

  (let ((*node-variables-tmp*)
	(*node-parameters-tmp*))
    (let ((body `(lambda ()
		   (let ((,(tensor-id toplevel) ,toplevel))
		     ,(trace-computation-node toplevel :forward)
		     ,(tensor-id toplevel)))))
      (when macroexpand
	(print body))

      ;; Enhancement: save the compiled body as fasl.
      (values (compile nil body)
	      (construct-variables-table (remove-duplicates *node-variables-tmp*))
	      (remove-duplicates *node-parameters-tmp*)))))

(defun construct-backward (out-scalar)

  )

(defun map-tree (fn tree)
  (let ((tree (funcall fn tree)))
    (if (listp tree)
        (mapcar (lambda (subtree)
                  (map-tree fn subtree))
                tree)
        tree)))

(defun dispatch-tensor-variable (form)
  (map-tree
   #'(lambda (x)
       (typecase x
	 (AbstractTensor
	  (if (eql (tensor-facet x) :input)
	      (push x *node-variables-tmp*))
	  (if (slot-value x 'requires-grad)
	      (push x *node-parameters-tmp*))
	  (tensor-id x))
	 ;; Add: AbstractNode
	 (T x)))
   form))

;; TODO: Use self
(defun trace-computation-node (toplevel
			       mode)
  (declare (type AbstractTensor toplevel)
	   (type (member :forward :backward) mode))
  (let ((state     (tensor-state toplevel))
	(variables (tensor-variables toplevel))
	(node      (tensor-backward toplevel)))
    (labels ((explore (var)
	       (trace-computation-node var mode)))
      (let ((next-states (loop for v in variables
			       if (tensor-state v)
				 collect (explore v)))
	    (node-id (gensym (format nil "~a" (class-name (class-of node))))))
	(case mode
	  (:forward
	   ;; current
	   ;; past
	   ;; Forward = reverse(build((car tensor)) + build((cdr tensor))) + ...
	   `(flet ((,node-id (,@(dispatch-tensor-variable variables))
		     ;; use state here, to avoid recomputing node.
		     ,(dispatch-tensor-variable (statecontainer-forward-out-form state))))
	      (let (,@(loop for v in variables collect `(,(tensor-id v) ,v)))
		,@next-states
		;; TODO: when 2nd forward, 3nd forward, ...? <- RESET Container
		(when (null (statecontainer-forward-result
			     (tensor-state ,(tensor-id toplevel))))
		  (setf
		   (statecontainer-forward-result
		    (tensor-state ,(tensor-id toplevel)))
		   (multiple-value-list (funcall #',node-id ,@(dispatch-tensor-variable variables)))))

		(setq ,(tensor-id toplevel)
		      (nth
		       ,(tensor-out-n toplevel)
		       (statecontainer-forward-result
			(tensor-state ,(tensor-id toplevel))))))))
	  (:backward
	   ;; past
	   ;; current

	   ))))))




(in-package :cl-waffe2/vm.generic-tensor)

;; CFFI-Styleの No Overhead generic-function
;; Building Forward/Backward codes

;;(defstruct (StateContainer)) result (計算結果), state, (実行されてない 実行された), form (S式)を格納できる

(defparameter *no-grad* nil "")

(defmacro with-no-grad (&body body)
  `(let ((*no-grad* t))
     ,@body))


(defstruct (StateContainer)
  (state :initialized :type (member :initialized :forwarded :backwarded))
  (forward-out-form nil :type list)
  (forward-result   nil :type list)

  (backward-input-variable)
  (backward-out-form nil :type list)
  (backward-result   nil :type list)

  (forward-n-out  0 :type fixnum)
  (backward-n-out 0 :type fixnum))

(defun state-reset! (tensor)
  "Resets tensor's result to get next round output."
  (declare (type AbstractTensor tensor))
  (setf (statecontainer-forward-result  (tensor-state tensor)) nil
	(statecontainer-backward-result (tensor-state tensor)) nil))

(defparameter *node-variables-tmp* nil)
(defparameter *node-parameters-tmp* nil)

(defun build (toplevel-tensor)
  "Return:
    (values forward backward variables parameters)"
  (multiple-value-bind (forward vars params) (construct-forward toplevel-tensor)
    (values
     forward
     (unless *no-grad*
       (construct-backward toplevel-tensor)
       nil)
     vars
     params)))

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
            all-tensors[list])"
  (declare (type AbstractTensor toplevel))

  ;; toplevel (usually out-scalar) -> forward -> each variables, parameters
  (let ((*node-variables-tmp*)
	(*node-parameters-tmp*))
    (let ((body `(lambda ()
		   ;;(declare (optimize (speed 3) (safety 0)))
		   (let ((,(tensor-id toplevel) ,toplevel))
		     ,(trace-forward-computation-node toplevel)
		     ,(tensor-id toplevel)))))
      (when macroexpand
	(print body))

      ;; Enhancement: save the compiled body as fasl.
      (values (compile nil body)
	      (construct-variables-table (remove-duplicates *node-variables-tmp*))
	      (remove-duplicates *node-parameters-tmp*)))))

(defun construct-backward (out-scalar &key (macroexpand nil))
  (declare (type AbstractTensor out-scalar))

  ;; out-scalar -> backward -> Each Parameters

  (let ((*node-variables-tmp*)
	(*node-parameters-tmp*))
    (let ((body
	    `(lambda ()
	       (let ((,(tensor-id out-scalar) ,out-scalar))
		 ,(trace-backward-computation-node out-scalar out-scalar)))))
      (when macroexpand
	(print body))
      (compile nil body))))

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
	  (push x *node-parameters-tmp*)
	  (tensor-id x))
	 ;; Add: AbstractNode
	 (T x)))
   form))

;; TODO: Use Self
(defun trace-forward-computation-node (toplevel)
  (declare (type AbstractTensor toplevel))
  (let ((state     (tensor-state toplevel))
	(variables (tensor-variables toplevel))
	(node      (tensor-backward toplevel)))
    (labels ((explore (var)
	       (trace-forward-computation-node var)))
      (let ((next-states (loop for v in variables
			       if (tensor-state v)
				 collect (explore v)))
	    (node-id (gensym (format nil "~a" (class-name (class-of node))))))
	;; current
	;; past
	;; Forward = reverse(build((car tensor)) + build((cdr tensor))) + ...
	`(flet ((,node-id (,@(dispatch-tensor-variable variables))
		  ;; use state here, to avoid recomputing node.
		  ,(dispatch-tensor-variable (statecontainer-forward-out-form state))))
	   (let (,@(loop for v in variables collect `(,(tensor-id v) ,v)))
	     ,@next-states
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
		     (tensor-state ,(tensor-id toplevel)))))))))))


;; TODO: NO GC, NO ALLOC
(defun trace-backward-computation-node (toplevel past-dy)
  (declare (type AbstractTensor toplevel))
  (let* ((state (tensor-state toplevel))
	 (variables  (tensor-variables toplevel))
	 (out-place (gensym "OUT"))
	 (dy-idx (tensor-id (statecontainer-backward-input-variable state)))
	 (node  (tensor-backward toplevel))
	 (node-id (gensym (format nil "~a" (class-name (class-of node))))))

    `(flet ((,node-id (,dy-idx)     
	      ,@(dispatch-tensor-variable
		 (statecontainer-backward-out-form state))))
       (let ((,out-place (multiple-value-list (,node-id ,(tensor-id past-dy)))))
	 (declare (ignorable ,out-place))
	 ,@(loop for k fixnum upfrom 0
		 for var in variables
		 if (tensor-state var)
		   collect (trace-backward-computation-node
			    var
			    `(nth ,k ,out-place))
		 if (slot-value var 'requires-grad)
		   collect `(set-grad `(nth ,k ,out-place) ,var))))))



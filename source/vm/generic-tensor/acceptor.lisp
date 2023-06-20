
(in-package :cl-waffe2/vm.generic-tensor)

(defparameter *no-grad* nil "TODO: DOC")

(defmacro with-no-grad (&body body)
  "TODO:DOC"
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

(defstruct (NodeVariables
	    (:constructor make-variable-table
		(symbols
		 variables
		 tmp-variables)))
  (symbols symbols :type list)
  (variables variables :type list)
  (tmp-variables tmp-variables :type list))

(defstruct (NodeParameters
	    (:constructor make-node-parameters (parameters)))
  (parameters parameters :type list))

(defmethod print-object ((node NodeVariables) stream)
  (let ((syms (nodevariables-symbols node))
	(vars (nodevariables-variables node))
	(table (make-print-table)))
    (format
     stream
     "+= [Computation Node Information] =======+

Subscripts:
~a

Variables
~a

 - The number of tmp variables: ~a
+========================================+" ;; TODO: Print Number of Parameters
     (with-output-to-string (out)
       (loop for k downfrom (length syms) to 0 by 2
	     if (nth k syms)
	       do (format out "     [~a -> ~a]~%" (nth k syms)
			  (or (nth (1+ k) syms) "?"))))
     
     (with-output-to-string (out)
       (let ((first-row)
	     (second-row))
	 (push "NAMES" first-row)
	 (push "SIZE"  second-row)
	 (loop for k upfrom 0 below (length vars) by 2
	       do (push (nth k vars) first-row)
		  (push (shape (nth (1+ k) vars)) second-row))
	 
	 (mapc #'(lambda (n s)
		   (addrow! table
			    (make-row `(,(format nil "~a" n)
					,(format nil "~a" s)))))
	       (reverse first-row) (reverse second-row))
	 (render-table table out)))
     (length (nodevariables-tmp-variables node)))))

(defmethod print-object ((params NodeParameters) stream)
  (format stream "#S(NODEPARAMETERS~%    :PARAMETERS (omitted)~%    :ntensors ~a)" (length (nodeparameters-parameters params))))

(defun embody-input (variables variable-name tensor)
  "(embody-input variables :a tensor)"
  (declare (type NodeVariables variables))

  (let ((input (getf (nodevariables-variables variables) variable-name)))
    (when (null input)
      (error "Couldn't find a variable of ~a" variable-name))

    (let ((symbols-changed (make-hash-table)))
      (loop for place in (tensor-input-shape input)
	    for val   in (shape tensor)
	    if (symbolp place)
	      do (setf (gethash place symbols-changed) val))

      ;; input <- tensor's visible-area
      (embody-actual-tensor input tensor)

      (maybe-optimize-memory-allocation
       variables
       symbols-changed))))

(defun maybe-delete-tensor-vec (tensor)
  (if (vec tensor)
      (tensor-delete tensor))
  (setf (tensor-vec tensor) nil))

(defun tensor-update-alloc (tensor name size)
  (declare (optimize (speed 3))
	   (type AbstractTensor tensor)
	   (type symbol name)
	   (type fixnum size))
  (let* ((shape    (the list (tensor-input-shape tensor)))
	 (name-pos (position name shape :test #'eql)))
    (when name-pos
      (setf (nth name-pos (slot-value tensor 'orig-shape)) size
	    (nth name-pos (slot-value tensor 'visible-shape)) size))))

(defun maybe-optimize-memory-allocation (variables symbols-changed)
  (declare (type NodeVariables variables))

  (let ((var-table (nodevariables-variables variables))
	(tmp-vars  (nodevariables-tmp-variables variables)))
    (maphash
     #'(lambda (key val)
	 ;; Detects the shape has changed compared with old-one.
	 (when (not (eql (getf var-table key) val))
	   (map 'list #'maybe-delete-tensor-vec tmp-vars)
	   (map 'list #'(lambda (x)
			  (tensor-update-alloc x key val))
		tmp-vars)))
     symbols-changed)
    t))

(defun state-reset! (tensor)
  "Resets tensor's result to get next round output."
  (declare (type AbstractTensor tensor))
  (if (tensor-state tensor)
      (setf (statecontainer-forward-result  (tensor-state tensor)) nil
	    (statecontainer-backward-result (tensor-state tensor)) nil)))

(defparameter *node-variables-tmp* nil)
(defparameter *node-parameters-tmp* nil)

;; fw, bw, vars, params = build(out-node, input-vars)
(defun build (toplevel-tensor
	      &key
		(ignore-optimize nil)
		(requires-grad (not *no-grad*))
		(macroexpand-forward nil)
		(macroexpand-backward nil))
  "Return:
    (values forward backward variables parameters)"

  (when (some #'symbolp (shape toplevel-tensor))
    (node-compile-error
     "Can't construct forward/backward code because the shape of toplevel-tensor should be determined by the end of nodes.

The shape of toplevel-tensor: ~a

Perhaps you're forgetting to call reducting APIs: !sum or !mean for example." (shape toplevel-tensor)))
  
  (multiple-value-bind (forward vars params backward)
      (construct-forward toplevel-tensor
			 :ignore-optimize ignore-optimize
			 :requires-grad requires-grad
			 :macroexpand-forward macroexpand-forward
			 :macroexpand-backward macroexpand-backward)
    (values
     forward
     backward
     vars
     (make-node-parameters params))))

(defun construct-variables-table (variables-list)
  "Returns variable-table where key and value is tensor's name and tensor's pointer."
  (declare (type list variables-list))
  (let ((variable-table `())
	(tmp-variable-table)
	(symbols `()))

    (mapc
     #'(lambda (tensor)
	 (let ((shapes (shape tensor)))
	   (loop for s in shapes
		 if (symbolp s)
		   do (setf (getf symbols s) nil)))

	 ;; tensor-name is keyword -> user-defined
	 ;; tensor-name is string  -> automatically-generated (assertion make not for users to never declare it)
	 
	 (if (typep (tensor-name tensor) 'keyword) ;; user-defined
	     (setf  (getf variable-table (tensor-name tensor)) tensor))

	 (if (typep (tensor-name tensor) 'string)
	     (push tensor tmp-variable-table)))
     variables-list)

    (make-variable-table symbols variable-table tmp-variable-table)))

(defun construct-forward (toplevel
			  &key
			    (requires-grad nil)
			    (macroexpand-forward nil)
			    (macroexpand-backward nil)
			    (ignore-optimize nil)
			    (unroll-threshold 10))
  "TODO: Docstring

Return:
    (values function-body[compiled-function]
            variables[plist]
            all-tensors[list])"
  (declare (type AbstractTensor toplevel))

  (unless ignore-optimize
    (optimize-computation-node! toplevel :speed 1))
  
  ;; toplevel (usually out-scalar) -> forward -> each variables, parameters
  (let ((*node-variables-tmp*)
	(*node-parameters-tmp*)
	(*unroll-threshold* unroll-threshold))
    (let ((body (compile-forward toplevel))
	  (backward (if requires-grad
			(construct-backward toplevel :macroexpand macroexpand-backward))))
      (let ((body `(lambda ()
		     (declare (optimize (speed 3)))
		     ;;(mapc #'result-reset! ',(remove-duplicates *node-parameters-tmp* :key #'tensor-vec))
		     (mapc #'state-reset! ',(remove-duplicates *node-parameters-tmp*))
		     (funcall ,body))))
	(when macroexpand-forward
	  (print body))

	;; Enhancement: save the compiled body as fasl.
	(values (compile nil body) 
		(construct-variables-table (remove-duplicates *node-variables-tmp*))
		(remove-duplicates *node-parameters-tmp*)
		backward)))))

(defun construct-backward (out-scalar
			   &key
			     (macroexpand nil))
  (declare (type AbstractTensor out-scalar))

  ;; out-scalar -> backward -> Each Parameters

  (let* ((out (if (scalar-p out-scalar)
		  (make-tensor 1
			       :dtype (dtype out-scalar)
			       :order (order out-scalar))
		  (make-tensor (shape out-scalar)
			       :dtype (dtype out-scalar)
			       :order (order out-scalar)
			       :initial-element 1)))
	 (body `(lambda ()
		  ,(explore-backwards out-scalar out)
		  t)))
    (when macroexpand (print body))
    (compile nil body)))

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
	  (tensor-id x))
	 ;; Add: AbstractNode
	 (T x)))
   form))

(defun compile-forward (toplevel)
  (declare (type AbstractTensor toplevel)
	   (optimize (speed 3)))
  (let ((state     (tensor-state     toplevel))
	(id        (tensor-id        toplevel))
	(variables (tensor-variables toplevel)))

    (when (or (detach-p toplevel) (null state))
      (return-from compile-forward (compile nil (lambda () toplevel))))

    (let ((next-states (map 'list #'compile-forward variables))
	  (out-places  (map 'list #'tensor-id       variables)))

      (compile nil
	       `(lambda ()
		  (declare (optimize (speed 3)))
		  ;; When Same Tensors continues, The variable hoge is defined but never used will be occurs...
		  (let* (,@(loop for f in next-states
				 for p in out-places
				 collect `(,p (funcall ,f)))
			 ,@(loop for v in (cl-waffe2/vm.nodes:node-local-variables (tensor-backward toplevel))
				 collect `(,(tensor-id v) ,v))
			 (,id ,toplevel))
		    (declare (type AbstractTensor ,@out-places)
			     (ignorable ,@out-places))
		    
		    ;; If toplevel isn't evaluated yet...
		    ,(and
		      (push toplevel *node-parameters-tmp*)
		      nil)
		    (when (null (statecontainer-forward-result (tensor-state ,id)))
		      (setf (statecontainer-forward-result (tensor-state ,id))
			    (multiple-value-list
			     (progn
			       ,(dispatch-tensor-variable (statecontainer-forward-out-form state))))))
		    
		    (nth ,(tensor-out-n toplevel) (statecontainer-forward-result (tensor-state ,id)))))))))

;; TODO
;; Most of backward-error, occurs here
;; So make it their log much clear.
;; Bug: Maybe this function doesn't work well.
(defun !maybe-move (place tensor &key (deterministic-p nil))
  "Moves the result of backwards, into variables where it was (if deterministic).

Return:
    (values next-tensor moved-p)
    moved-p ... If p, the tensor should be moved to the variable."
  ;; If deterministic-p = t, do in-place.
  (with-no-grad
    (when tensor
      (if (movetensor-p (tensor-backward tensor))
	  (values tensor nil);; the tensor is already copied?
	  (cl-waffe2/vm.nodes:with-shape-checkpoint (:moving nil)
	    (let ((place (if deterministic-p
			     place ;; place=tensor.variables[n]
			     (make-tensor (if (scalar-p place)
					      0
					      (shape place))
					  :dtype (dtype place)
					  :order (order place)))))
	      ;; Forcibly moving them.
	      (values (cl-waffe2/base-impl:!move place tensor) deterministic-p)))))))

;; there remains to be improved (in term of compile-speed)
(defun explore-backwards (toplevel past-dy)
  "Constructs the computation node for backwards."
  (declare (type AbstractTensor toplevel past-dy))
  
  ;; g(t's dout dx dy dz...) -> [t-1]'s dout
  ;; first past-dy = 1.0

  ;; Get a backward node.

  ;; ((self dout dx dy)
  ;;   (values (kernel1) (kernel2)))
  ;; Step1. dx.dy copy Step2 kernel2, kernel2, Step3 Move
  ;; copy -> compile -> backward -> copy


  ;; The problem is that:
  ;; (values (!move dx (!mul dout dy)) (!move dx (!mul dout dy)))
  ;; produces side effects
  
  (when (null (tensor-backward toplevel))
    (return-from explore-backwards))

  ;; Record: at what node, the backward error was occured.
  (cl-waffe2/vm.nodes:with-shape-checkpoint (:backward (tensor-backward toplevel))
    ;; In order to restore tensors' backwards, keep them saving at backwards-tmp.
    (let* ((ancestor-p-map (map 'list #'ancestor-param-p (tensor-variables toplevel)))
	   (deterministic-n (count t ancestor-p-map :test #'(lambda (x y) (and x y))))
	   (deterministic-p (<= deterministic-n 1))
	   ;; Explore previous backwards of tensor-variables.
	   (outs (apply
		  ;; (backward self dout dx dy dz ...)
		  #'cl-waffe2/vm.nodes:backward
		  (tensor-backward toplevel)		  
		   past-dy;;(detach! past-dy)
		  (map 'list #'detach! (tensor-variables toplevel))))
	   (moved-p-list)
	   ;; Pruning unused nodes by exploring ancestor-param-p
	   ;; dy1 <- dx.grad
	   (outs (loop for var in (tensor-variables toplevel)
		       for out in outs
		       for k upfrom 0
		       if (ancestor-param-p var)
			 collect
			 (multiple-value-bind (res moved-p)
			     (!maybe-move var out
					  :deterministic-p (or deterministic-p
							       (= k (1- deterministic-n))))
			   (push moved-p moved-p-list)
			   res)
		       else
			 collect nil
		       finally (setq moved-p-list (reverse moved-p-list))))
	   (movers (loop for var in (tensor-variables toplevel)
			 for out in outs
			 for mv? in moved-p-list
			 if (and out mv?)
			   collect
			   (prog1 ;; var <- out
			       (let ((*no-grad* t)) (compile-forward (cl-waffe2/base-impl:!move (detach! var) (detach! out))))
			     (setf (detach-p var) nil
				   (detach-p out) nil)))))

      ;; dx1 <- dx'
      ;; dy1 <- dy'

      ;; dx <- dx1
      ;; dy <- dy1
		     
      ;; all backwards' node should ends with MoveTensorNode
      #|
      (assert (every (compose #'movetensor-p #'tensor-backward)
		     outs)
	      nil
      "Explore-Backwards: Assertion Failed because the nodes: ~a aren't ended with MoveTensorNode/MoveTensorScalarNode" (tensor-variables toplevel))
      |#

      (labels ((expand-backward-values (variables outs paramp body)
		 (cond
		   ((and (null variables)
			 (null outs))
		    body)
		   ((and (car variables) (car outs) (car paramp))
		    `(let ((,(tensor-id (car outs)) (funcall ,(compile-forward (car outs)))))
		       ,(expand-backward-values (cdr variables) (cdr outs) (cdr paramp) body)))
		   (T (expand-backward-values (cdr variables) (cdr outs) (cdr paramp) body)))))

	;; Body
	`(let*-ignorable
	     (,@(map 'list
		     #'(lambda (tensor)
			 `(,(tensor-id tensor) ,tensor))
		     (tensor-variables toplevel))
	      ,@(loop for tensor in outs
		      if tensor
			collect `(,(tensor-id tensor) ,tensor)))
	   ,(expand-backward-values
	     (tensor-variables toplevel)
	     outs
	     (map 'list #'(lambda (x) (slot-value x 'requires-grad)) (tensor-variables toplevel))
	     `(progn
		,@(map 'list #'(lambda (v x) `(setq ,(tensor-id v) (funcall ,x)))
		       (tensor-variables toplevel) movers)
		,@(loop for v  in (tensor-variables toplevel)
			for o  in outs			
			if (slot-value v 'requires-grad)
			  collect `(add-grads ,v ,(tensor-id o))
			if (and ;;(not (slot-value v 'requires-grad))
			    o
			    (tensor-state v))
			  collect (progn
				    (setf (detach-p v) nil)
				    (setf (detach-p o) nil)
				    (explore-backwards v o))))))))))

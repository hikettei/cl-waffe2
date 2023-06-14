
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

(defmethod print-object ((node NodeVariables) stream)
  (let ((syms (nodevariables-symbols node))
	(vars (nodevariables-variables node)))
    (format
     stream
     "{NodeVariables
    - symbols
~a
    - variables
~a
    - :n-tmp-variables ~a}"
     (with-output-to-string (out)
       (loop for k upfrom 0 below (length syms) by 2
	     do (format out "     [~a -> ~a]~%" (nth k syms) (nth (1+ k) syms))))
     (with-output-to-string (out)
       (loop for k upfrom 0 below (length vars) by 2
	     do (format out "     [~a -> ~a]~%" (nth k vars) (shape (nth (1+ k) vars)))))
     (length (nodevariables-tmp-variables node)))))

(defstruct (NodeParameters))

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
     params)))

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
    
    (when (null state)
      (return-from compile-forward #'(lambda () toplevel)))

    (let ((next-states (map 'list #'compile-forward variables))
	  (out-places  (map 'list #'tensor-id       variables)))

      (compile nil
	       `(lambda ()
		  (declare (optimize (speed 3)))
		  ;; When Same Tensors continues, The variable hoge is defined but never used will be occurs...
		  (let* (,@(loop for f in next-states
				 for p in out-places
				 collect `(,p (funcall ,f)))
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
(defun !maybe-move (place tensor)
  "If previous node of tensor is MoveTensor, this function is ignored, otherwise do a !move"
  (when tensor
    (if (movetensor-p (tensor-backward tensor))
	tensor
	(cl-waffe2/vm.nodes:with-shape-checkpoint (:moving nil)
	    (cl-waffe2/base-impl:!move place tensor)))))

(defun explore-backwards (toplevel past-dy)
  "Constructs the computation node for backwards."
  (declare (type AbstractTensor toplevel past-dy))
  ;; g(t's dout dx dy dz...) -> [t-1]'s dout
  ;; first past-dy = 1.0

  ;; Get a backward node.

  (when (null (tensor-backward toplevel))
    (return-from explore-backwards))

  ;; Record: at what node, the backward error was occured.
  (cl-waffe2/vm.nodes:with-shape-checkpoint (:backward (tensor-backward toplevel))
    (let* ((backwards-tmp (map 'list #'tensor-backward (tensor-variables toplevel)))
	   (outs (apply
		  ;; (backward self dout dx dy dz ...)
		  #'cl-waffe2/vm.nodes:backward
		  (tensor-backward toplevel)
		  past-dy
		  ;; detach from computation node by projecting into -> <t>.
		  (map 'list #'detach! (tensor-variables toplevel))))
	   (outs (map 'list #'!maybe-move (tensor-variables toplevel) outs)))

      ;; should ends with MoveTensorNode
      #|
      (assert (every (compose #'movetensor-p #'tensor-backward)
		     outs)
	      nil
      "Explore-Backwards: Assertion Failed because the nodes: ~a aren't ended with MoveTensorNode/MoveTensorScalarNode" (tensor-variables toplevel))
      |#
      
      ;; FixME: (!add k k) produces style-warning.
      `(let* (,@(map 'list
		     #'(lambda (tensor)
			 `(,(tensor-id tensor) ,tensor))
		     (tensor-variables toplevel))
	      ,@(loop for tensor in outs
		      if tensor
			collect `(,(tensor-id tensor) ,tensor)))
	 (declare (ignorable ,@(map 'list #'tensor-id (tensor-variables toplevel)))
		  (ignorable ,@(map 'list #'tensor-id (loop for o in outs if o collect o))))
	 
	 ,@(loop for out    in outs
		 for tensor in (tensor-variables toplevel)
		 for bw in backwards-tmp
		 if out
		   collect `(let ((,(tensor-id out) (funcall ,(compile-forward out))))
			      (declare (ignorable ,(tensor-id out)))
			      ,(if (slot-value tensor 'requires-grad)
				   ;; !copy and add.
				   `(add-grads ,(tensor-id tensor) ,(tensor-id out))
				   (when (and
					  (tensor-state tensor)
					  (ancestor-param-p tensor))
				     ;; Explore deeper if there's any params.
				     (setf (tensor-backward tensor) bw)
				     (explore-backwards tensor out)))))))))


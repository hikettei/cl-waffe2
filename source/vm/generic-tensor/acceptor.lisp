
(in-package :cl-waffe2/vm.generic-tensor)

(defparameter *no-grad* nil "If t, all operations don't create gradients.")

(defmacro with-no-grad (&body body)
  "
## [macro] with-no-grad

```lisp
(with-no-grad &body body)
```

Set `*np-grad*` `t` under the `body` execution, no gradients are made for backward.
"
  `(let ((*no-grad* t))
     ,@body))


;; StateContainer is a temporary node for creating forward/backward
(defstruct (StateContainer)
  (state :initialized :type (member :initialized :forwarded :backwarded))
  (forward-out-form nil :type function)
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

(defvar *node-parameters-tmp* nil)

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

(defun map-tree (fn tree)
  (let ((tree (funcall fn tree)))
    (if (listp tree)
        (mapcar (lambda (subtree)
                  (map-tree fn subtree))
                tree)
        tree)))

(defun register-variables (list)
  (dolist (tensor list)
    (when (typep tensor 'AbstractTensor)
      (unless (find tensor *node-parameters-tmp* :test #'equal)
	(push tensor *node-parameters-tmp*)))))

;; ==============================================================================
;; Kernel Constructor
;; ==============================================================================


(defun compile-forward-chain (toplevel &key (read-save-for-backward nil))
  "
## [function] compile-forward-chain

"
  (declare (type AbstractTensor toplevel))
  
  (when (or (detach-p toplevel) (null (tensor-state toplevel)))
    (return-from compile-forward-chain
      (if read-save-for-backward
	  `(or (read-save-for-backward ,toplevel) ,toplevel)
	  toplevel)))
  
  (let* ((state (tensor-state toplevel))
	 (vars  (tensor-variables toplevel))
	 (fw-compiled (statecontainer-forward-out-form state)))
    

    (let ((next-states (map 'list #'(lambda (x) (compile-forward-chain x :read-save-for-backward read-save-for-backward)) vars))
	  (out-places  (map 'list #'tensor-id vars)))

      ;; Tensors to use is determined when compiled
      ;; InputTensor ... Symbols(A, B) is changed, alloc is changed.

      (register-variables out-places)
      (register-variables (tensor-variables toplevel))
      
      `(let*-ignorable (,@(loop for s in next-states ;; p <- s
				for p in out-places
				collect `(,p ,s))
			,@(loop for v in (cl-waffe2/vm.nodes:node-local-variables (tensor-backward toplevel))
				collect `(,(tensor-id v) ,v))
			(,(tensor-id toplevel) (progn ,toplevel)))

	 ;; The Operation hasn't done yet...
	 (when (null (statecontainer-forward-result (tensor-state ,(tensor-id toplevel))))
	   (setf (statecontainer-forward-result (tensor-state ,(tensor-id toplevel)))
		 (multiple-value-list (funcall ,fw-compiled ,@(map 'list #'tensor-id vars)))))

	 (nth ,(tensor-out-n toplevel) (statecontainer-forward-result (tensor-state ,(tensor-id toplevel))))))))

(defun compile-backward-chain (toplevel past-dy)
  "Constructs the computation node for backwards."
  (declare (type AbstractTensor toplevel past-dy))
    
  (when (null (tensor-backward toplevel))
    (return-from compile-backward-chain))

  ;; Record: at what node, the backward error was occured.
  (cl-waffe2/vm.nodes:with-shape-checkpoint (:backward (tensor-backward toplevel))
    ;; In order to restore tensors' backwards, keep them saving at backwards-tmp.
    (let* ((outs (apply
		  ;; (backward self dout dx dy dz ...)
		  #'cl-waffe2/vm.nodes:backward
		  (tensor-backward toplevel)	  
		  (detach! past-dy)
		  (map 'list #'detach! (tensor-variables toplevel))))
	   (out-kernels (map 'list #'second outs))
	   (next-dys    (map 'list #'first  outs))
	   (movers      (map 'list #'third outs)))

      (setf (detach-p past-dy) nil)
      `(let (,@(loop for out-kernel in out-kernels
		     for ndy in next-dys
		     if out-kernel
		       collect `(,(tensor-id ndy) ,ndy;;(funcall ,out-kernel ,(tensor-id past-dy))
				 )))
	 (declare (ignorable ,@(loop for out-kernel in out-kernels
				     for ndy in next-dys
				     if out-kernel
				       collect (tensor-id ndy))))

	 (let (,@(loop for mv in movers
		       for dy in next-dys
		       if mv
			 collect `(,(tensor-id dy) (funcall ,mv))))
	   (declare (ignorable ,@(loop for mv in movers
				       for dy in next-dys
				       if mv
					 collect (tensor-id dy))))

	   ;; dx.grad += grad
	   ,@(loop for next-dy in next-dys
		   for var in (tensor-variables toplevel)
		   if (slot-value var 'requires-grad)
		     collect `(add-grads ,var ,(tensor-id next-dy))
		   if (and next-dy
			   (tensor-backward var)
			   (ancestor-param-p var))
		     collect (compile-backward-chain var next-dy)))))))


;; Toplevel
(defun compile-forward-kernel (toplevel &key (read-save-for-backward nil))
  "
## [function] compile-forward-kernel
"
  (optimize-computation-node! toplevel :speed 1)
  (let ((*node-parameters-tmp*))
    (let ((body (compile-forward-chain toplevel :read-save-for-backward read-save-for-backward)))
      ;; (declare (optimize (speed 3) (safety 0)))
      (values
       (compile nil
		`(lambda ()
		   (map 'list #'state-reset! ',*node-parameters-tmp*)
		   ,body))
       *node-parameters-tmp*))))

(defun compile-backward-kernel (toplevel)
  (let* ((out (if (scalar-p toplevel)
		  (make-tensor 1
			       :dtype (dtype toplevel)
			       :order (order toplevel))
		  (make-tensor (shape toplevel)
			       :dtype (dtype toplevel)
			       :order (order toplevel)
			       :initial-element 1)))
	 (body `(lambda ()
		  (let ((,(tensor-id out) ,out))
		    (declare (ignorable ,(tensor-id out)))
		    ,(compile-backward-chain toplevel out)
		    t))))
    (compile nil body)))



;; ==========================================
;; Rewriting
;; ==========================================


(defclass Compiled-Composite ()
  ((compiled-forward :initarg :compiled-forward :type function :reader compiled-forward)
   (compiled-backward :initarg :compiled-backward :type (or null function) :reader compiled-backward)
   (variables :initarg :variables :reader compiled-variables)
   (parameters :initarg :parameters :reader compiled-parameters)
   (first-call-p :initform nil :accessor composite-first-call-p :type boolean)))

(defmethod cl-waffe2/vm.nodes:forward ((model Compiled-Composite) &rest inputs &key &allow-other-keys)
  ;; Keywords :A A :B B ...
  ;; forward
  (funcall (compiled-forward model)))

(defmethod cl-waffe2/vm.nodes:backward ((model Compiled-Composite) &rest inputs)
  (funcall (compiled-backward model)))

(defun build (toplevel
	      &key (construct-backward? (not *no-grad*)))

  (when (some #'symbolp (shape toplevel))
    (error "Can't construct forward, because the shape of tensor is undetermined: ~a" (shape toplevel)))
  
  (multiple-value-bind (forward-kernel vars) (compile-forward-kernel toplevel)
    ;; Vars - All Variables (including ChainTMP) used in forward.
    (make-instance 'Compiled-Composite
		   :variables  (construct-variables-table vars)
		   :parameters (make-node-parameters
				(loop for v in vars
				      if (slot-value v 'requires-grad)
					collect v))
		   :compiled-forward forward-kernel
		   :compiled-backward (when construct-backward?
					(compile-backward-kernel toplevel)))))

;; TODO: Print-Obj
;; TODO  Fix Tests
;; TODO  Embody Input
;; TODO  Optimizing constructing backward (いらないところまでTraceしてそ~~)
;; TODO 
(defmethod print-object ((model Compiled-Composite) stream)
  (format stream "<Compiled-Composite
    forward:  ~a
    backward: ~a
   (TODO)
>"
	  ;; Variables
	  (compiled-forward model)
	  (compiled-backward model)))

;; set variable

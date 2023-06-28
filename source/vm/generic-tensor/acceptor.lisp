
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


;; StateContainer is a structure which accompany with Tensors
;; and is used to share:
;; Forward/Backward Forms
;; Their computation results
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
		(parameters
		 symbols
		 adjustable-symbol-table
		 variables
		 tmp-variables)))
  (parameters parameters :type list)
  (symbols       symbols :type list)   ;; 'a -> current-size 'b -> current-size
  (adjustable-symbol adjustable-symbol-table :type hash-table) ;; 'a -> max-alloc-size 'b -> max-alloc-size
  (variables     variables :type hash-table) ;; (make-input `(...) :A)
  (tmp-variables tmp-variables :type list))  ;; (make-input `(...) nil) i.e.: chaintmp

(defmethod print-object ((node NodeVariables) stream)
  (let* ((syms  (nodevariables-symbols node))
 	 (vars  (nodevariables-variables node)) ;; Hash-Table
	 (input-keys (alexandria:hash-table-keys vars))
	 (table (make-print-table)))
    (format
     stream
     "+= [NodeVariables Information] =======+

Subscripts:
~a

Variables:
~a

 - The number of tmp variables : ~a
 - The number of parameters    : ~a
+========================================+" ;; TODO: Print Number of Parameters
     (with-output-to-string (out)
       (loop for k downfrom (length syms) to 0 by 2
	     if (nth k syms)
	       do (format out "     [~a -> ~a, max=~a]~%"
			  (nth k syms)
			  (or (nth (1+ k) syms) "?")
			  (or (gethash (nth k syms) (nodevariables-adjustable-symbol node)) "?"))))

     (with-output-to-string (out)
       (let ((first-row)
	     (second-row))
	 (push "NAMES" first-row)
	 (push "SIZE"  second-row)
	 (loop for k upfrom 0 below (length input-keys)
	       do (push (nth k input-keys) first-row)
		  (push (shape (gethash (nth k input-keys) vars)) second-row))
	 
	 (mapc #'(lambda (n s)
		   (addrow! table
			    (make-row `(,(format nil "~a" n)
					,(format nil "~a" s)))))
	       (reverse first-row) (reverse second-row))
	 (render-table table out)))
     (length (nodevariables-tmp-variables node))
     (length (nodevariables-parameters node)))))


;; Rewrite
(defun embody-input (nodevars variable-name actual-tensor)
  "(embody-input variables :a tensor)"
  (declare (type NodeVariables nodevars))
  
  (let ((input-tensor (gethash variable-name (nodevariables-variables nodevars))))

    (when (null input-tensor)
      (error "The InputTensor named ~a weren't appeared in the computation node" variable-name))

    (let ((symbols-changed (make-hash-table)))
      (loop for place in (tensor-input-shape input-tensor)
	    for value in (shape actual-tensor)
	    if (and (not (symbolp place))
		    (not (= place value)))
	      do (error "Can't embody ~a into fixed place, ~a.
~a and ~a"
			value place input-tensor actual-tensor)
	    if (symbolp place)
	      do (setf (gethash place symbols-changed) value))

      ;; Checking if the new size never beyonds memory-pool.

      (let ((maxsize (nodevariables-adjustable-symbol nodevars)))
	(maphash
	 #'(lambda (key value)
	     (let ((max-val (gethash key maxsize)))
	       (when (and (not (null max-val))
			  (> value max-val))
		 (error "Error: Can't embody tensor because ~a = ~a is given but ~a must <= ~a"
			key
			value
			key
			max-val))

	       (when (null max-val)
		 (setf (gethash key maxsize) value))))
	 symbols-changed))
      
      ;; InputTensor <- Actual-Tensor
      (embody-actual-tensor input-tensor actual-tensor)

      ;; Apply hash-table

      (maphash
       #'(lambda (key value)
	   (setf (getf (nodevariables-symbols nodevars) key) value))
       symbols-changed)

      t)))

(defun state-reset! (tensor)
  "Resets tensor's result to get next round output."
  (declare (type AbstractTensor tensor))
  (if (tensor-state tensor)
      (setf (statecontainer-forward-result  (tensor-state tensor)) nil
	    (statecontainer-backward-result (tensor-state tensor)) nil)))

(defvar *node-parameters-tmp* nil "An temporary variable to store all the variables used in the computation node.")

(defun construct-variables-table (variables-list)
  ;; variables-list = *node-parameters-tmp*
  "Returns variable-table where key and value is tensor's name and tensor's pointer."
  (declare (type list variables-list))
  (let ((variable-table (make-hash-table))
	(tmp-variable-table) ;; All the InputTensor
	(adjustable-symbol-table (make-hash-table))
	(parameters `())
	(symbols `()))

    (mapc
     #'(lambda (tensor)
	 (let ((shapes (shape tensor)))
	   (loop for s in shapes
		 if (symbolp s)
		   do (setf (gethash s adjustable-symbol-table) nil)
		      (setf (getf symbols s) nil)))

	 ;; tensor-name is keyword -> user-defined
	 ;; tensor-name is string  -> automatically-generated (assertion make not for users to never declare it)

	 ;; (make-input ... :Train-X)
	 (when (slot-value tensor 'requires-grad)
	   (push tensor parameters))
	 
	 (when (user-input-p tensor)
	   (setf (gethash (tensor-name tensor) variable-table) tensor))

	 ;; (make-input ... ChainTMPXXX)
	 (when (typep (tensor-name tensor) 'string)
	   (push tensor tmp-variable-table)))
     variables-list)

    (make-variable-table
     parameters
     symbols
     adjustable-symbol-table
     variable-table
     tmp-variable-table)))

(defun map-tree (fn tree)
  (let ((tree (funcall fn tree)))
    (if (listp tree)
        (mapcar (lambda (subtree)
                  (map-tree fn subtree))
                tree)
        tree)))

(defun register-variables (list)
  ;; Registers Parameter/Variables If any.
  (dolist (tensor list)
    (when (typep tensor 'AbstractTensor)
      (unless (find tensor *node-parameters-tmp* :test #'equal)
	  (push tensor *node-parameters-tmp*)))))

;; ==============================================================================
;; Kernel Constructor
;; ==============================================================================

(defun compile-forward-chain (toplevel)
  "
## [function] compile-forward-chain

Tracing until one of variables reached a toplevel tensor (detach-p is t or no backwards), returning an S-expression which can be compiled.
"
  (declare (type AbstractTensor toplevel))

  (when (or (detach-p toplevel) (null (tensor-state toplevel)))
    (return-from compile-forward-chain toplevel))

  
  (let* ((state (tensor-state toplevel))
	 (vars  (tensor-variables toplevel))
	 (fw-compiled (statecontainer-forward-out-form state)))

    (let ((next-states (map 'list #'compile-forward-chain vars))
	  (out-places  (map 'list #'tensor-id vars)))
      
      ;;
      ;; f(x, y)
      ;; x <- x.func
      ;; y <- y.func
      ;; f(x.func, y.func)

      ;; Tensors to use is determined when compiled
      ;; InputTensor ... Symbols(A, B) is changed, alloc is changed.

      (register-variables out-places)
      (register-variables (tensor-variables toplevel))

      ;; Declare undetermined shapes
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
  "Constructs the computation node for backwards recursively."
  (declare (type AbstractTensor toplevel past-dy))
  
  (when (null (tensor-backward toplevel))
    (return-from compile-backward-chain))

  ;; with-shape-checkout: at where node, the backward error was occured?
  (cl-waffe2/vm.nodes:with-shape-checkpoint (:backward (tensor-backward toplevel))
    ;; In order to restore tensors' backwards, keep them saving at backwards-tmp.

    ;; The backward function is: g(dout) -> x.grad, y.grad where/dx/dy is a constant parameter. dout is a variable.
    (let* ((outs (apply
		  ;; (backward self dout dx dy dz ...)
		  ;; -> (backward self dout)
		  #'cl-waffe2/vm.nodes:expand-backward
		  ;; Here, we trace the definition of backward.
		  (tensor-backward toplevel)
		  past-dy
		  (tensor-variables toplevel))))

      ;; Rewrite
      
      ;; Memo: All backward nodes, are ends with MoveTensorNode
      `(let (,@(loop for var in (tensor-variables toplevel)
		     for kernel in outs
		     if kernel
		       collect `(,(tensor-id var) (funcall (the function ,kernel) ,(tensor-id past-dy)))))
	 (declare (ignorable ,@(loop for var in (tensor-variables toplevel)
				     for kernel in outs
				     if kernel collect (tensor-id var))))
	 ;; Explore deeper, or ,if any, add grads to the parameter
	 ,@(loop for var in (tensor-variables toplevel)
		 for kernel in outs
		 if (slot-value var 'requires-grad)
		   collect `(add-grads ,var ,(tensor-id var))
		 if (and kernel
			 (tensor-backward var)
			 (ancestor-param-p var))
		   collect (compile-backward-chain var var))))))


;; Toplevel
(defun compile-forward-kernel (toplevel
			       &key
				 (compile-mode :fastest))
  "
## [function] compile-forward-kernel
"

  (declare (type compile-option-t compile-mode))
  ;; Pruning unused nodes.
  (optimize-computation-node! toplevel :speed 1)
  
  (let* ((*node-parameters-tmp*)
	 (body (compile-forward-chain toplevel))
	 (inputs (loop for var in *node-parameters-tmp*
		       if (user-input-p var)
			 collect var))
	 (set-input-forms))

    (mapc #'(lambda (input)
	      (loop for shape in (shape input)
		    for kth-dim upfrom 0
		    if (symbolp shape)
		      do (push `(',shape (nth ,kth-dim (shape ,input))) set-input-forms)))
	  inputs)
    
    (values
     (compile nil
	      `(lambda ()
		 (declare ,(compile-option-form compile-mode))
		 (map 'list #'state-reset! ',*node-parameters-tmp*)
		 (with-adjustable-symbols (,@set-input-forms)
		   ,body)))
     *node-parameters-tmp*)))

;; TODO: In order to backward with make-input, expand with-adjustable-symbols is needed. <- Do it at toplevel
(defun make-vm-function (toplevel)
  (optimize-computation-node! toplevel :speed 1)
  
  (let ((*node-parameters-tmp*))
    (compile-forward-chain toplevel)))


(defun compile-backward-kernel (toplevel &key (compile-mode :fastest))
  (declare (type compile-option-t compile-mode))
  (let* ((out (if (scalar-p toplevel)
		  (make-tensor 1
			       :dtype (dtype toplevel)
			       :order (order toplevel))
		  (make-tensor (shape toplevel)
			       :dtype (dtype toplevel)
			       :order (order toplevel)
			       :initial-element 1)))
	 (body `(lambda ()
		  ;; TODO: with-adjustable-symbols, with-no-grad,
		  (let ((,(tensor-id out) ,out))
		    (declare (ignorable ,(tensor-id out))
			     ,(compile-option-form compile-mode))
		    (let ((*no-grad* t))
		      ,(compile-backward-chain toplevel out))
		    t))))
    (compile nil body)))



;; ==========================================
;; Rewriting
;; ==========================================


(defclass Compiled-Composite ()
  ((compiled-forward :initarg :compiled-forward :type function :reader compiled-forward)
   (compiled-backward :initarg :compiled-backward :type (or null function) :reader compiled-backward)
   (variables :initarg :variables :reader compiled-variables)
   (first-call-p :initform nil :accessor composite-first-call-p :type boolean)))

(defmethod cl-waffe2/vm.nodes:forward ((model Compiled-Composite) &rest inputs &key &allow-other-keys)
  ;; Keywords :A A :B B ...
  ;; forward
  (when inputs
    (warn "forward: Inputs are ignored"))
  
  (funcall (compiled-forward model)))

(defmethod cl-waffe2/vm.nodes:backward ((model Compiled-Composite) &rest inputs)

  (when inputs
    (warn "backward: Inputs are ignored"))
  (funcall (compiled-backward model)))

(defmethod set-input ((model Compiled-Composite) input-name actual-value)
  "
## [method] set-input

"
  (embody-input (compiled-variables model) input-name actual-value))

(defmethod get-input ((model Compiled-Composite) input-name)
  "
## [method] get-input

"
  (gethash input-name (nodevariables-variables (compiled-variables model))))

(defun build (toplevel
	      &key (construct-backward? (not *no-grad*)))

  (when (some #'symbolp (shape toplevel))
    (error "Can't construct forward, because the shape of tensor is undetermined: ~a" (shape toplevel)))
  
  (multiple-value-bind (forward-kernel vars) (compile-forward-kernel toplevel)
    ;; Vars - All Variables (including ChainTMP) used in forward.
    (make-instance 'Compiled-Composite
		   :variables  (construct-variables-table vars)
		   :compiled-forward forward-kernel
		   :compiled-backward (when construct-backward?
					(compile-backward-kernel toplevel)))))

;; TopLevelでシンボルを初期化 (OK)
;; tensor-vec/memory-poolを更新

;; TODO:
;;
;; 1. Memory-Pool/Tensor-vecを続きやる
;; 2. Forwardを正しく行う
;; 3. 適切な場所でSave-For-Backwardが呼び出されているか？Copyが重複していないか？
;;
(defmethod print-object ((model Compiled-Composite) stream)
  (format stream "<Compiled-Composite
    forward:  ~a
    backward: ~a

~a
>"
	  ;; Variables
	  (compiled-forward model)
	  (compiled-backward model)
	  (compiled-variables model)))

;; set variable

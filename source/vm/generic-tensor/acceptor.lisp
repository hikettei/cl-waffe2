
(in-package :cl-waffe2/vm.generic-tensor)

;;
;; acceptor.lisp provides an compiler of given nodes, and general-purpose APIs to handle with cl-waffe2 nodes.
;;

(defparameter *no-grad* nil "If t, no gradients are made for backwards.")

(defparameter *calling-backward-mode* nil)

(defmacro with-no-grad (&body body)
  "
## [macro] with-no-grad

```lisp
(with-no-grad &body body)
```

Under the `body` execution, the macro sets `*no-grad*` = `t`, that is, the built nodes are regarded as: no gradients are made for backwards.

"
  `(let ((*no-grad* t))
     ,@body))


;; StateContainer is a structure which accompany with Tensors
;; and is used to share:
;; Forward/Backward Forms
;; Their computation results
(defstruct (StateContainer)
  (state :initialized :type (member :initialized :forwarded :backwarded))
  (forward-out-form nil :type Compiled-Kernel)
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
     "+= [Tensors in the computation node] =======+

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

;; ↓ All fixed, nevermind.
;; ==============================================================================================================================
;; [FixME] Someone destructs (tensor-input-shape tensor) in toplevel (destructs = permute* shuffles the order of it)
;; I tried harder, but I couldn't find which function shuffles it, but I've confirmed other slots (strides, visible-shape, views...) are NEVER affected.
;; I know this solution is literally UGLY, but decided to create a LUT (*shape-input-table*, id->first input-shape), and ignore
;; destructed input-shape.
;; This sometime may result an unexcepted behaviour, but:
;; tensor-id is created by (gensym), so assured that never conflicts.
;; (i.e.: Once the tensor(id=A) is initialized as `(A B), no one can create another tensor whose id=A)
;; As long as we don't add any strange features, I think we'll be fine.
;; One of my concern is that: when allocating a cache tensor for InputTensor, we must not use input-shape.
;; ==============================================================================================================================

;;(defparameter *shape-input-table* (make-hash-table) "Things to be deleted in the future release.")

(defun embody-input (nodevars variable-name actual-tensor)
  "(embody-input variables :a tensor)

InputTensor[A x B] <- ExistTensor[10 x 10] for example.
See also: `set-input`"
  (declare (type NodeVariables nodevars))
  
  (let ((input-tensor (gethash variable-name (nodevariables-variables nodevars))))
    
    (when (null input-tensor)
      (error "The InputTensor named ~a weren't appeared in the computation node" variable-name))

    (when (not (= (dims input-tensor) (dims actual-tensor)))
      (error "embody-input: ranks does not match: ~a and ~a" input-tensor actual-tensor))
    
    (let ((symbols-changed (make-hash-table)))
      (loop for place in (tensor-input-shape input-tensor)
	    for value in (shape actual-tensor)
	    for rank upfrom 0
	    if (and (not (symbolp place))
		    (not (= place value)))
	      do (error "embody-input: The ~ath rank is a fixed dimension.
So the corresponding shape must be ~a but got ~a.

Shapes: Input_Place <- Actual_Tensor
-----------------------------------------------
Shapes:   ~a~a  <-  ~a

input-tensor:
~a

Strides       : ~a
permute-order : ~a

actual-tensor:
~a

Strides       : ~a
permute-order : ~a
"
			rank value place
			variable-name
			(tensor-input-shape input-tensor)
			(shape actual-tensor)
			input-tensor
			(tensor-stride input-tensor)
			(tensor-permute-order input-tensor)
			actual-tensor
			(tensor-stride actual-tensor)
			(tensor-permute-order actual-tensor))
	    if (symbolp place)
	      do (setf (gethash place symbols-changed) value))
      
      ;; Checking if the new size never beyonds memory-pool.

      ;; No need to check.
      
      #|
      (let ((maxsize (nodevariables-adjustable-symbol nodevars)))
	(maphash
	 #'(lambda (key value)
	     (declare (ignore value))
	     (let ((max-val (gethash key maxsize)))
	       
	       (when (and (not (null max-val))
			  ;;(> value max-val)
			  )
		 
		 ;;(error "Error: Can't embody tensor because ~a = ~a is given but ~a must <= ~a"
		 ;;	key
		 ;;	value
		 ;;	key
		 ;;	max-val)
		 ;;(setf (gethash key maxsize) value)
		 )
	       

	       (when (null max-val)
		 ;; (setf (gethash key maxsize) value)
		 )))
      symbols-changed))
      |#
      
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

(defun state-reset-bw! (tensor)
  "Resets tensor's result to get next round output."
  (declare (type AbstractTensor tensor))
  (if (tensor-state tensor)
      (setf (statecontainer-backward-result (tensor-state tensor)) nil)))

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

(defun declare-compiled-composite (model)
  "Extends into *node-parameters-info*, about the given model's variable informations."
  (declare (type Compiled-Composite model))
  (let ((variables-info (compiled-variables model)))
    (register-variables
     (nodevariables-tmp-variables variables-info))
    (register-variables
     (nodevariables-parameters variables-info))
    nil))

(defparameter *runtime-shape-inspection* nil)

;; ==============================================================================
;; Kernel Constructor | General-Purpose APIs
;; ==============================================================================

(defun runtime-shape-inspection (declared-shape tensor)
  (declare (type list declared-shape)
	   (type AbstractTensor tensor)
	   (optimize (speed 3)))

  (shape-equal-list declared-shape (shape tensor)))

;; TODO: read-result should be inlined
(declaim (inline read-result)
	 (ftype (function (AbstractTensor) AbstractTensor) read-result))
(defun read-result (tensor)
 "Returns the result of computing of tensor in the compiled code"
  (declare (type AbstractTensor tensor))

  (let ((state (tensor-state tensor)))
    (if state
        (nth (tensor-out-n tensor) (the list (statecontainer-forward-result (tensor-state tensor))))
	tensor)))

;; Set *runtime-shape-inspection* = t to detect run-time shape-error
(defun compile-forward-chain (toplevel
			      &key
				(stop-me nil)
				(called-with-vars nil))
  "
## [function] compile-forward-chain

Tracing until one of variables reached a toplevel tensor (detach-p is t or no backwards), returning an S-expression which can be compiled by processing systems.
"
  (declare (type AbstractTensor toplevel))

  (when (or stop-me (null (tensor-state toplevel)))
    (return-from compile-forward-chain toplevel))

  (when (detach-p toplevel)
    ;; After reading 
    (setq stop-me t))
  
  (let* ((state (tensor-state toplevel))
	 (vars  (tensor-variables toplevel))
	 (node  (tensor-backward toplevel))
	 (fw-compiled (statecontainer-forward-out-form state)))

    (when (compiled-kernel-cache-p fw-compiled)
      (push fw-compiled *kernel-storeroom*))
    
    (let ((next-states (map 'list #'(lambda (x) (compile-forward-chain x :stop-me stop-me :called-with-vars toplevel)) vars)))
      
      ;;
      ;; f(x, y)
      ;; x <- x.func
      ;; y <- y.func
      ;; f(x.func, y.func)

      ;; Tensors to use is determined when compiled
      ;; InputTensor ... Symbols(A, B) is changed, alloc is changed.

      
      (register-variables (tensor-variables toplevel))

      `(progn
	 
	 ;; The Operation hasn't done yet:
	 ;; The code below seems ugly...
	 (when (or (null (statecontainer-forward-result (tensor-state ,toplevel)))
		   (when *calling-backward-mode*
		     (let ((out (statecontainer-backward-result (tensor-state ,toplevel))))
		       (car out))))

	   ;; Seems ugly:
	   ;; Judge the tensor is created when forward time or backward time
	   (when (and *calling-backward-mode*
		      (not (car (statecontainer-backward-result (tensor-state ,toplevel)))))
	     (setf (statecontainer-backward-result (tensor-state ,toplevel))
		   (list (null (statecontainer-forward-result (tensor-state ,toplevel))))))


	   ;; Calls an event: on-finalizing-compiling
	   ;; If JIT is implemented by user, expand user defined forms
	   
	   (setf (statecontainer-forward-result (tensor-state ,toplevel))
		 (multiple-value-list (call-kernel ,fw-compiled ,@(loop for arg in next-states collect arg)))))

	 ,(when (tensor-backward toplevel)
	    (cl-waffe2/vm.nodes:on-finalizing-compiling
	     (tensor-backward toplevel)
	     toplevel
	     called-with-vars
	     nil))
	 

	 ;; TODO UPDATE
	 ,(when (and node
		     *runtime-shape-inspection*)
	    `(unless *calling-backward-mode*
	       (assert
		(runtime-shape-inspection
		 (nth ,(tensor-out-n toplevel) ',(cl-waffe2/vm.nodes:node-output-shape node)) ;; Declared output shape
		 (nth ,(tensor-out-n toplevel) (statecontainer-forward-result (tensor-state ,toplevel)))) ;; Output shape compiled
		nil
		"Assertion Failed.: Detected Shape-Error in runtime.
At      : ~a
Returned: ~a
Declared: ~a

The definition/implementation of nodes could be invaild."
		,node
		(nth ,(tensor-out-n toplevel) ',(cl-waffe2/vm.nodes:node-output-shape node))
		(shape (nth ,(tensor-out-n toplevel) (statecontainer-forward-result (tensor-state ,toplevel))))))) ;; Output shape compiled
	 
	 (nth ,(tensor-out-n toplevel) (statecontainer-forward-result (tensor-state ,toplevel)))))))

(defun compile-backward-chain (toplevel past-dy)
  "Constructs the computation node for backwards recursively."
  (declare (type AbstractTensor toplevel past-dy))

  (when (null (tensor-backward toplevel))
    ;; Gradient-add-form here?
    (return-from compile-backward-chain
      (when (slot-value toplevel 'requires-grad)
	(init-optimizer-utils! toplevel)
	;; We receive gradients as vec permuted (see MoveTensorNode)
	`(add-grads ,toplevel ,(tensor-id past-dy)))))

  ;; Not anymore used.
  #|
	(if t;(not (permuted-p past-dy))
	    `(add-grads ,toplevel ,(tensor-id past-dy))
	    (progn
	      ;; The function gradient-adder is compiler for default permutation tensors
	      ;; So if the gradients are given as permuted, we need to recompile gradient-adder.
	      (setf (gradient-adder toplevel)
		    (make-gradient-adder
		     toplevel
		     (shape toplevel)
		     :use-input
		     past-dy))
	      ;; X.grad += permute*(value, 0 1 ...)
	      ;; ^ recompiling is needed!
	      `(add-grads ,toplevel ,(tensor-id past-dy)))))))
 |#

  ;; with-shape-checkout: at where node, the backward error was occured?
  (cl-waffe2/vm.nodes:with-shape-checkpoint (:backward (tensor-backward toplevel))
    ;; In order to restore tensors' backwards for a future printing error place, keep them saving at backwards-tmp.

    ;; The backward function is: g(dout) -> x.grad, y.grad where/dx/dy is a constant parameter. dout is a variable.
    (let* ((outs (apply
		  ;; (backward self dout dx dy dz ...)
		  ;; -> (backward self dout)
		  #'cl-waffe2/vm.nodes:expand-backward
		  ;; Here, we trace the definition of backward.
		  (tensor-backward toplevel)
		  past-dy
		  (tensor-variables toplevel)))
	   (next-dys   (map 'list #'car outs))
	   (outs       (map 'list #'cdr outs)))

      ;; tensor-id used here never conflicts
      ;; because each time backward goes deeper, the new tensor-id is generated.
      `(let (,@(loop for kernel in outs
		     for out in next-dys
		     for var in (tensor-variables toplevel)
		     if (and kernel (ancestor-param-p var))
		       ;; g_x(dout_past, x_next) -> dout_next, g_y ...
		       collect `(,(tensor-id out) (funcall (the function ,kernel) ,(tensor-id past-dy)))))
	 (declare (ignorable ,@(loop for o in next-dys
				     for v in (tensor-variables toplevel)
				     if (and o (ancestor-param-p v))
				       collect (tensor-id o))))
	 ;; Explore deeper, or ,if any, add grads to the parameter
	 ,@(loop for var in (tensor-variables toplevel)
		 for kernel in outs
		 for next-dy in next-dys
		 if (and kernel
			 (ancestor-param-p var))
		   collect (compile-backward-chain var next-dy))))))

;; Toplevel
;; This is not for users.
(defun compile-forward-kernel (forward-iseq
			       variables
			       &key
				 (compile-mode :default))
  "
## [function] compile-forward-kernel
"

  (declare (type compile-option-t compile-mode)
	   (ignore compile-mode))
  ;; Pruning unused nodes.
  ;;(optimize-computation-node! toplevel :speed 1)
  
  (let* ((inputs (loop for var in variables
		       if (user-input-p var)
			 collect var))
	 (set-input-forms))
    ;; set-input-form .. collects adjustable shapes
    (mapc #'(lambda (input)
	      (loop for shape in (shape input)
		    for kth-dim upfrom 0
		    if (symbolp shape)
		      do (push `(',shape (nth ,kth-dim (shape ,input))) set-input-forms)))
	  inputs)

    (values
     (compile nil
	      `(lambda ()
		 (with-adjustable-symbols (,@set-input-forms)
		   (cl-waffe2/vm:accept-instructions ',forward-iseq))))
     variables
     set-input-forms)))

;; TODO: In order to backward with make-input, expand with-adjustable-symbols is needed. <- Do it at toplevel
(defun make-vm-function (toplevel)
  (optimize-computation-node! toplevel :speed 1)

  `(progn
     ,(compile-forward-chain toplevel)))

(defun compile-backward-kernel (toplevel backward-iseq &key (compile-mode :default) (set-input-forms))
  (declare (type compile-option-t compile-mode)
	   (ignore compile-mode))

  
  (when (some #'symbolp (shape toplevel))
    (error "Can't construct backward, because the shape of tensor is undetermined: ~a

Try again with: (with-no-grad ...) " (shape toplevel)))
  
  (let* ((body (if set-input-forms
		   `(with-adjustable-symbols (,@set-input-forms)
		      (cl-waffe2/vm:accept-instructions ',backward-iseq))
		   `(cl-waffe2/vm:accept-instructions ',backward-iseq))))
    (compile nil `(lambda () ,body t))))

;; ==========================================
;; General-Purpose APIs
;; ==========================================

;; In acceptor.lisp:
;; Compiled-Composite forward backward set-input get-input is all you need!

(defclass Compiled-Composite ()
  ((compiled-forward :initarg :compiled-forward :type function :reader compiled-forward)
   (compiled-backward :initarg :compiled-backward :type (or null function) :reader compiled-backward)
   (variables :initarg :variables :reader compiled-variables)
   (first-call-p :initform nil :accessor composite-first-call-p :type boolean))
  (:documentation "
## [class] Compiled-Composite

Compiled-Composite is a `callable` CLOS class, and holds compiled forward/backward function of all the computation node to all the endpoints from the top of the models' neural network. Also, this class holds information of all variables used in the node.

It is NOT possible to construct a computation node after Compiled-Composite, If you need this, try consider using the function `cl-waffe2/base-impl:proceed`.

The class will appear in your project with calling the function `build`, set the toplevel node (e.g.: the result of criterion when the task is optimizing.) to the first argument. cl-waffe2 compiler will instantly construct an lambda function of forward/backward, which is invoked by calling `(forward compiled-composite)` or `(backward compiled-composite)` method.

See also: `build` `set-input` `get-input`.

### Examples

(TODO)

"))

(defun all-embodied? (model)
  "Invokes an simple-error when model has still unembodied tensor"
  (declare (type Compiled-Composite model))
  (let ((vars (nodevariables-variables (compiled-variables model)))
	(unembodied))
    (loop for key being the hash-keys in vars using (hash-value val)
	  if (null (vec val))
	    do (push (cons key val) unembodied))

    (when unembodied
      (error "Can't call forward of the given model,
because there's still unembodied tensors:
~a"
	     (with-output-to-string (out)
	       (dolist (k unembodied) (format out "~% :~a - ~a Tensor." (car k) (shape (cdr k)))))))))


(defmethod cl-waffe2/vm.nodes:forward ((model Compiled-Composite) &rest inputs &key &allow-other-keys)
  (when inputs
    (warn "forward: Inputs for compiled-composite are ignored"))

  ;; Check if all the inputs are embodied?
  (all-embodied? model)

  (let ((*runtime-mode-p* t))
    (funcall (compiled-forward model))))

(defmethod cl-waffe2/vm.nodes:backward ((model Compiled-Composite) &rest inputs)

  (when inputs
    (warn "backward: Inputs for compiled-composite are ignored"))
  (when (null (compiled-backward model))
    (error "cl-waffe2/vm.nodes:backward. Because the model was compiled with (with-no-grad ) mode, a backward function wasn't compiled."))

  (let ((*runtime-mode-p* t))
    (funcall (compiled-backward model))))

(defmethod set-input ((model Compiled-Composite) input-name actual-value)
  "
## [method] set-input

```
(set-input (model Compiled-Composite) input-name actual-value)
```

Embodies an `InputTensor` in the model. All unembodied tensors in the model can be accessed by printing the model.

`input-name` could be a keyword indicating input-tensor, `actual-value` is a `AbstractTensor` whose facet = `:exist` (created by `make-tensor`).


"
  (embody-input (compiled-variables model) input-name actual-value))

(defmethod get-input ((model Compiled-Composite) input-name)
  "
## [method] get-input

```
(get-input (model Compiled-Composite) input-name)
```

Reading all variables in the computation node, the method get-input returns an corresponding `InputTensor` of model.

"
  (gethash input-name (nodevariables-variables (compiled-variables model))))

(defun build (toplevel
	      &key
		(construct-backward? (not *no-grad*))
		(compile-mode :fastest)
		(use-setinput-form nil))
  "
## [function] build

```lisp
(build toplevel
	      &key
		(construct-backward? (not *no-grad*))
		(compile-mode :fastest))
```

Receiving the toplevel node in the neural network, the function `build` constructs a optimal forward/backward function, returning `Compiled-Composite`.

### The constraints of toplevel tensor.

The shape of topleve mustn't include a `symbol`.

For example, this cl-waffe2 operation is invaild. because the function `(!sin x)` still returns `(A B)` tensor.

```lisp
(build (!sin (make-input `(A B) :Input)))
```

In order to build this operation correctly, calling `criterion` (intrinsically, `!sum` or `!mean`) is a vaild option for neural network tasks.

```lisp
(build (!sum (!sin (make-input `(A B) :input)))) ;; Passes Correctly!
```

After working with adjustable shape tensor, don't forget to embody the InputTensor!

```lisp
(let ((compiled-model (build (!sum (!sin (make-input `(A B) :input))))))
    (set-input compiled-model :input (randn `(10 10)))
    (forward compiled-model))
```

### Inputs

`toplevel [AbstractTensor]` the end of nodes. for neural network tasks, this should be scalartensor or tensors with total elements is 1, but since cl-waffe2 is intended to be applied other tasks, cl-waffe2 never produce warning while other frameworks like PyTorch will return error if `<<(10 10)Tensor>>.backward()` is invoked for example.

`construct-backward?` [boolean] If t, the backward construction won't be done.

`compile-mode`[compile-mode-t] an keyword to indicate compiling option.
"
  (declare (type AbstractTensor toplevel))

  (multiple-value-bind (fw-iseq bw-iseq variables) (cl-waffe2/vm:compile-forward-and-backward toplevel :need-backward construct-backward?)
    (multiple-value-bind (fw-function variables set-input-forms) (compile-forward-kernel fw-iseq variables :compile-mode compile-mode)
      ;; Vars - All Variables (including ChainTMP) used in forward.
      (prog1
	  (values (make-instance 'Compiled-Composite
				 :variables  (construct-variables-table variables)
				 :compiled-forward fw-function
				 :compiled-backward (when construct-backward?
						      (compile-backward-kernel toplevel bw-iseq :compile-mode compile-mode :set-input-forms set-input-forms)))
		  
		  (when use-setinput-form set-input-forms))
	(mapc #'cl-waffe2/vm.nodes:on-finished-compiling *using-backend*)))))

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



(in-package :cl-waffe2/vm.nodes)
  
(defclass AbstractNode ()
  ((local-variables :accessor node-local-variables :type list :initform nil)
   (function-node
    :initarg
    :function-node
    :reader abstractnode-node
    :type function) ;; [x ~ y] [y z] -> [z x]
   (function-node1
    :initarg
    :function-node1
    :reader abstractnode-node1
    :type (or null function)) ;; [x y] [y z] -> [z x], ~ is removed. If ~ isn't used at function-node, set nil
   (uprank-state :initform nil :initarg :uprank-state :reader uprank-state :type list)
   (transmission-state :initarg :transmission-state :reader transmission-state :type list)
   (ignore-shape-error :initform nil :accessor ignore-shape-error)
   (passed-at-least-once :initform nil :accessor node-passed-p :type boolean))
  (:documentation "The class AbstractNode is a fundamental object of describing computation nodes in cl-waffe.

AbstractNode must possess following:

   1. Transimission State

   2. Slots (for passing forward/backward)

   3. Variables (for building computation nodes)
"))

;; Unused?
(defmethod test-and-forward-shape ((node AbstractNode) &rest previous-shape)
  ""
  (funcall (abstractnode-node node) previous-shape))

(defun describe-problem-at (error-node inputs outputs &aux (saved-state (checkpoint-node-at *shape-error-when*)))
  (case (checkpoint-state *shape-error-when*)
    (:forward
     (format
      nil
      "Couldn't step forward because of shape-error.

The operation was : ~a

Input(s)            : ~a
Predicted Output(s) : ~a"
      error-node
      (map 'list #'shape inputs)
      outputs))
    (:backward
     (format
      nil
      "Shape-Error was detected during backward construction.

When : building backward for ~a

The operation was   : ~a

Input(s)            : ~a
Predicted Output(s) : ~a"
      saved-state
      error-node
      (map 'list #'shape inputs)
      outputs))
    (:moving
     (format
      nil
      "Attempted to construct backward, but the shape of inputs and gradients do not match.

When : building backward for ~a
The function backward returned a tensor with the shape of ~a.
However, it should be ~a.

== [Repeating The Same Contents] ==============================================
The operation was: ~a
"
      ;; Saved-State should be MoveTensorNode...
      saved-state
      (shape (second inputs))
      (shape (car    inputs))
      error-node))))

(defun describe-problems (error-node detected-errors inputs outputs)
  "Creates a report of shape-error"
  ;; Enhancement:
  ;; Restart-Case
  ;; [Fix-Definition-And-Step]
  ;; [Replace-Shape-And-Step]
  ;; More Details:
  ;; Displays [pre-|post-]computation node
  ;; TODO: make it more intuitive....
  (shaping-error
   "~a

Here's a list of reports.

1. ~a

~a
~a"
   (describe-problem-at error-node inputs outputs)
   (car detected-errors)
   (if (cdr detected-errors)
       "Also, these reports could be helpful for you (calculated ignoring the first errors.)"
       "")
   (with-output-to-string (out)
     (loop for err in (cdr detected-errors)
	   for n upfrom 2
	   do (format out "~%~%~a. ~a" n err)))))

(defun make-grad-gensym ()
  (intern (symbol-name (gensym "Chain")) "KEYWORD"))

;; Forward:  f(input-state) -> output-state
;; Backward: g(output-state) -> input-state

(defgeneric forward  (node &rest inputs))
(defgeneric backward (node &rest inputs))

;; Enhancement: !t is matmul-dedicated, therefore (!add (!t x) y) is invaild.
;; Enhancement: A[~] -> B[~] <- replace A with input-name.
(defmethod forward :around ((node AbstractNode) &rest inputs)
  ;; Update Computation Nodes

  ;; TODO: Put warning when !t without matmul
  (let* ((transition-function     (abstractnode-node node))  ;; original subscript
	 (transition-function-sub (abstractnode-node1 node)) ;; subscript without ~
	 (pointer-states          (transmission-state node)) ;; <- what ptr/view to use?
	 (uprankable-list (uprank-state node))
	 (input-states (loop for i in inputs collect (shape i)))
	 ;; Records that Is it worth to trace backward?
	 (ancestor-param-p (some #'cl-waffe2/vm.generic-tensor:ancestor-param-p inputs)))

    ;; Input-State -> Output-State
    (multiple-value-bind (out-state detected-errors) (funcall transition-function input-states)
      ;; FixME: ~ = nil isn't allowed. [~ x] with (10) is unexceptedly invaild.

      (when detected-errors
	;; If any errors occured, try again with removing ~ from subscripts. (I know this behaviour is ugly.)

	(multiple-value-bind (out-state1 detected-errors-1) (funcall transition-function-sub input-states)
	  
	  ;; Enhancement
	  ;; CALL-VIEW-AND-CONTINUE
	  ;; If error is not originated from ~.

	  ;; The case when error continues...

	  (if (and detected-errors-1
		   (not (ignore-shape-error node)))
	      ;; There's no flexible tensor, then it is invaild.
	      ;; If there's any flexible tensor, uprank it and try again.

	      ;; The node is declared as uprankable
	      ;;  A[~ x]   B[x]   -> B[x]
	      ;; flexible normal
	      (if (find t (mapcar #'(lambda (x y)
				      (and (tensor-flexible-p x) y))
				  inputs uprankable-list))
		  ;; Update ranks and try again...
		  (let* ((largest-axis (loop for i in input-states
					     for tensor in inputs
					     unless (tensor-flexible-p tensor)
					       maximize (length i)))
			 (largest-axis
			   (if (= largest-axis 0) ;; every inputs are flexible
			       (loop for i in input-states
				     maximize (length i))
			       largest-axis))
			 (largest-axis-shape
			   (shape
			    (find largest-axis inputs
				  :test #'=
				  :key #'(lambda (x) (length (shape x)))))))
		    ;; The :where is...
		    ;; [~ x y] <- it is ok to apply uprank rule.
		    ;; [x y]   <- it is ng to apply uprank rule.
		    (return-from forward
		      (apply
		       #'forward
		       node
		       (loop for input in inputs
			     for uprankable in uprankable-list
			     if (and (tensor-flexible-p input)
				     uprankable)
			       collect (let* ((rankup-n (- largest-axis (length (shape input))))
					      (out (cl-waffe2/base-impl:!rankup
						    input
						    rankup-n))
					      (subscripts (loop for i upfrom 0 below rankup-n
								collect `(:broadcast ,(nth i largest-axis-shape))))
					      (out (apply #'cl-waffe2/base-impl:!view out subscripts)))
					 ;; Apply Broadcast to flexible axis
					 (setf (tensor-flexible-p out) nil)
					 out)
			     else
			       collect input))))
		  ;; Otherwise the operation was invaild.
		  (describe-problems node detected-errors inputs out-state))
	      (setq out-state out-state1))))

      ;; TODO: When Dynamic-Mode
      ;; Call (construct-forward) and eval it here.
      
      ;; Forward:  Input-State  -> Output-State
      ;; Backward: Output-State -> Input-State

      ;; Forward:
      ;; [30 10] + [30 10] -> [10 10] -> [5 5]
      ;; Memo: Sharing Allocated memory between f and b
      ;; can be realised with self ...
      ;; recompute grad

      (let* ((forward-form (call-next-method))
	     (next-tensor
	       (loop for shape in out-state
		     for nth-arg upfrom 0
		     for extend-from in pointer-states
		     ;; Make -> ScalarTensor if shape = (1)
		     collect (let* ((next-tensor
				      (make-input shape nil
						  :scalar-p (out-scalar-p node)
						  :dtype (dtype (nth (or extend-from 0) inputs))
						  :order (order (nth (or extend-from 0) inputs))))
				    (state (make-statecontainer
					    :forward-out-form forward-form
					    :forward-n-out  (length out-state)
					    :backward-n-out (length input-states))))

			       ;; Extend Views, Strides, Orig-Shapes, etc..

			       (when extend-from
				 ;; Detect Errors
				 (let ((input (nth extend-from inputs)))
				   (setf (slot-value next-tensor 'cl-waffe2/vm.generic-tensor::orig-shape)
					 (slot-value input       'cl-waffe2/vm.generic-tensor::orig-shape)
					 
					 (tensor-view next-tensor)
					 (tensor-view input)
					 
					 (slot-value next-tensor 'cl-waffe2/vm.generic-tensor::projected-p)
					 (slot-value input 'cl-waffe2/vm.generic-tensor::projected-p)
					 
					 (cl-waffe2/vm.generic-tensor:tensor-stride next-tensor)
					 (cl-waffe2/vm.generic-tensor:tensor-stride input))


				   (when (cl-waffe2/vm.generic-tensor::vec input)
				     (setf (tensor-vec next-tensor) (tensor-vec input)
					   (cl-waffe2/vm.generic-tensor::tensor-facet input) :exist))))
			       
			       (setf (cl-waffe2/vm.generic-tensor:ancestor-param-p next-tensor) ancestor-param-p)
			       (setf (tensor-out-n next-tensor)     nth-arg)
			       (setf (tensor-state next-tensor)     state)
			       (setf (tensor-backward next-tensor)  node)
			       (setf (tensor-variables next-tensor) inputs)
			       next-tensor))))
	(apply #'values next-tensor)))))

(defmethod forward ((node AbstractNode) &rest inputs)
  (declare (ignore inputs))
  ;; Describe More Errors.
  (error "Couldn't step forward because ~a forward is undefined.

Make sure that the node has been initialised using the constructor automatically generated by the defnode macro.

(DO NOT USE make-instance for defnode) but use:

(~a &rest inputs).

In cl-waffe, AbstractNode (i.e.: nodes defined by defnode itself), doesn't have a definition of forward and backward.
Use the define-impl macro to give definitions for the node and forward them.
"
	 node
	 (class-name (class-of node))))


;; Inputs: dout dx dy dz ...
;;       InputTensor dx dy dz ...
(defmethod backward :around ((node AbstractNode) &rest inputs)
  (when (not *no-grad*)
    (with-no-grad
      (with-shape-checkpoint (:backward node)
	(let* ((dout (car inputs))
	       (moveplist)
	       (out-kernels (multiple-value-list (call-next-method)))
	       (out-kernels (loop for out in out-kernels
				  for in  in (cdr inputs)
				  for k upfrom 0
				  if out
				    collect (multiple-value-bind (res mv)
						(!maybe-move in out :deterministic-p (= (1- (length (cdr inputs))) k))
					      (push mv moveplist)
					      res)
				  else
				    collect (and (push nil moveplist) nil)
				  finally (setq moveplist (reverse moveplist))))
	       (compiled-g (map 'list #'(lambda (x) (when x (cl-waffe2/vm.generic-tensor:make-vm-function x :read-save-for-backward t))) out-kernels))
	       (movers (loop for v   in (cdr inputs)
			     for out in out-kernels
			     for mv? in moveplist
			     if (and out mv?)
			       collect (prog1
					   ;; Var <- Var.Grad
					   (cl-waffe2/vm.generic-tensor:make-vm-function (cl-waffe2/base-impl:!move (detach v t) (detach out t)))
					 (detach v nil)
					 (detach out nil))
			     else
			       collect nil)))
	  ;; dout should be make-input
	  ;; g(dout, dx, dy, ..., dn) -> dx.grad, gy.grad, ..., dn.grad
	  (loop with dout-real = (gensym "dout")
	        for g in compiled-g
		for o in out-kernels
		for m in movers
		;; g(x) * n
		if g
		  collect
		  (list
		   o
		   `(lambda (,dout-real)
		      (with-no-grad
			(cl-waffe2/vm.generic-tensor:embody-actual-tensor ,dout ,dout-real)
			;; print (g)
			(funcall ,g)))
		   m)
		else
		  collect nil))))))

(defmethod backward ((node AbstractNode) &rest inputs)
  (declare (ignore inputs))
  (error "Couldn't step backward because ~a backward is undefined." node))



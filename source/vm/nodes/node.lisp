
(in-package :cl-waffe2/vm.nodes)

;; TODO: Error Printings
(defparameter *enable-broadcasting-auto* t) ;; This parameter never exported/modified by users, just used to prevent recursively call of forward
(defparameter *restart-variable-from* nil)

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
   (excepted-output-shape :initform nil :type list :accessor node-output-shape)
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

(defun parse-broadcasted-shape (shapes)
  (flet ((apply-refine (list)
	   (loop for s in list
		 unless (and *enable-broadcasting-auto* ;; not created by broadcast
			     (equal s -1))
		   collect (if (equal s -1)
			       1
			       s))))
    (map 'list #'apply-refine shapes)))

;; Optim:
;;
;; (3 -1 3)
;; (3 3 -1)
;;
;; (-1 3 3)
;;    (3 3)
;;

			       
;; ugh... steadly it gets ugly...
;; Enhancement: !t is matmul-dedicated, therefore (!add (!t x) y) is invaild.
;; Enhancement: A[~] -> B[~] <- replace A with input-name.
(defmethod forward :around ((node AbstractNode) &rest inputs)
  ;; Update Computation Nodes

  ;; TODO: Put warning when !t without matmul

  (let* ((save-for-backward (node-save-for-backward node))
	 (inputs (or (when *restart-variable-from* inputs) ;; No additional save-for-backward is created.
		     (loop for i in inputs
			   for k upfrom 0
			   if (and
			       (not *no-grad*)
			       (nth k save-for-backward)) ;; If T?
			     collect (let ((*enable-broadcasting-auto* nil)) (system-lazy-set-save-for-backward i))
			   else
			     collect i)))
	 (transition-function     (abstractnode-node node))  ;; original subscript
	 (transition-function-sub (abstractnode-node1 node)) ;; subscript without ~
	 (pointer-states          (transmission-state node)) ;; <- what ptr/view to use?
	 (uprankable-list (uprank-state node))
	 ;; replace <1 x N> = -1 for instance
	 (input-states (map 'list #'shape inputs))
	 
	 ;; Records that Is it worth to trace backward?
	 (ancestor-param-p (some #'cl-waffe2/vm.generic-tensor:ancestor-param-p inputs)))
    ;; Detecting Shape-Error, And finds combinations that satisfies shape-requirement heuristic.
    ;; Input-State -> Output-State
    (multiple-value-bind (out-state detected-errors) (funcall transition-function input-states)
      ;;(setq out-state (delete-broadcast out-state))
      ;; FixME: ~ = nil isn't allowed. [~ x] with (10) is unexceptedly invaild.

      (when detected-errors
	;; If any errors occured, try again with removing ~ from subscripts. (I know this behaviour is ugly.)

	(multiple-value-bind (out-state1 detected-errors-1) (funcall transition-function-sub input-states)
	  ;;(setq out-state1 (delete-broadcast out-state1))

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

	      ;; Uprankable Nodes are subject to broadcasted
	      (if (and *enable-broadcasting-auto*
		       (find t (mapcar #'(lambda (x y) (and (tensor-flexible-p x) y)) inputs uprankable-list)))
		  ;; Update ranks and try again...
		  (let* ((*enable-broadcasting-auto* nil)
			 (inputs-new (apply-broadcast input-states inputs uprankable-list))
			 (*restart-variable-from* inputs)) ;; (tensor-variable out) records the first call of tensors (top_inputs)
		    ;;  Forward:         Broadcast:          Restart
		    ;; 
		    ;; top_inputs -> View/Reshape/Uprank -> (forward ...)
		    ;; inputs-top -> inputs-new nodes are continuous.
		    ;; because inputs-new are made from inputs-top
		    (return-from forward (apply #'forward node inputs-new)))
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

      (setq out-state (parse-broadcasted-shape out-state))
      (setf (node-output-shape node) out-state)

      (let* ((forward-form (call-next-method))
	     (next-tensor
	       (loop for shape in out-state
		     for nth-arg upfrom 0
		     for extend-from in pointer-states
		     ;; Make -> ScalarTensor if shape = (1)
		     collect (let* ((next-tensor
				      (make-input shape nil
						  :create-from (when extend-from
								 (nth extend-from inputs))
						  :scalar-p (out-scalar-p node)
						  :dtype (dtype (nth (or extend-from 0) inputs))
						  :order (order (nth (or extend-from 0) inputs))))
				    (state (make-statecontainer
					    :forward-out-form forward-form
					    :forward-n-out  (length out-state)
					    :backward-n-out (length input-states))))

			       ;; Extend Views, Strides, Orig-Shapes, etc..
			       ;; Exntend Permuted Stride Orders

			       (when extend-from
				 ;; FixME: A[i j] -> A[j i] is invaild beucase before and after the operation, indicates the same pointer but shapes arenot the same.
				 ;; Detect Errors
				 (let ((input (nth extend-from inputs)))
				   ;; Extend View Forms
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


;; cl-waffe2 backward semantics:
;; It is nothing but a topdown AD. (Comments below is just for myself.)
;; There's a five kinds of operations used in deep-learning (for convinience I call it so)
;; 1. f(x) (e.g.: sin/cos etc...)
;; 2. f(x, y) (e.g.: axpy add sub)
;; 3. f_swap(x, y) (e.g.: gemm. mul, div, save-for-backward=t)
;; 4. Matmul
;; 5. View/Broadcsting (e.g.: !view !sum !flexible)
;;
;; == [Memo] ================================================
;;
;; Keep in-place operation for f_swap(x, y) backward (e.g.: Mul/DivNode), with numerical stability?
;;
;; MulTensorNode with Parameter Argument, requires 3 times copy:
;; 1. x_clone to avoid being parameter destructed
;; 2. x_save_for_backward to compute backward.
;;
;; g(dout, x_input, y_input) = MulTensorNode.backward
;; = Move(x_place, dout*y_input), Move(y_place, dout*x_input)
;;
;; (If x_place were x_input, the second operation won't performed well because x_input is invaild.)
;;
;; where x_input is 
;; 1. x_save_for_backward If corresponding save-for-backward is t.
;; 2. x                   otherwise (being destructed by other node, just cache place.)
;; 
;; where x_place is
;; 1. Node.variables[0]
;;
;; Memo: Separate x_place and x_input
;; ==========================================================
;;
;; in-place operation for f(x, y) backward (e.g.: Axpy)
;; g(dout, x_in, y_in) = AxpyNode.backward
;; = Move(x_place, dout), Move(y_place, dout)
;; Here, x_place/y_place never share the same pointer, that is, it works as a brunching.
;;
;; ==========================================================
;;
;; one-arg function, f(x) (e.g.: SinNode)
;; g(dout, x_in) = Move(x_place, cos(x_in)) where x_in = x_save_for_backward, x_place is SinNode.variables[0]
;;
;; ==========================================================
;;
;; MoveTensorNode's backward
;;
;; !move is defined as: Move(place, target)
;; g(dout, x_in, y_in) = (nil, Move(y_place, y_in))
;;
;; y_in = (previous dout)
;; y_place =
;; 1. Copy of Node.variables[0] If the argument is Parameter/ExistTensor, which is NEVER allowed to modify.
;; 2. ChainTMP otherwise
;;

(defun select-return-place (place argn nth-trying)
  (if (or ;;(tensor-projected-p place)
	  (not (= argn nth-trying)))
      (make-input (shape place) nil
		  :create-from place
		  :dtype (dtype place)
		  :order (order place)
		  :scalar-p (scalar-p place))
      place))


(defun adjust-bw-place (bw-node place argn nth-trying)
  "If the bw-node ends with MoveTensorNode, return itself, otherwise add MoveTensorNode."
  
  (when bw-node
    (if (movetensor-p (tensor-backward bw-node))
	bw-node
	(with-shape-checkpoint (:moving nil)
	  (let ((out (cl-waffe2/base-impl:!move
		      (select-return-place place argn nth-trying)
		      bw-node
		      :force t)))
	    
	    ;; F(x, y, ...)
	    ;; x.state = :chain / :input?

	    (if (eql (cl-waffe2/vm.generic-tensor::tensor-attribute place) :chain)
		out
		;; when bw-node... -> chain not connected well...?
		bw-node)))))) ;; Make-copy

(defun expand-backward (node dout &rest inputs-out)
  "
## [function] expand-backward

Constructs an backward function of the given node, with inputs.

Returns an lambda-function(dout) (not being compiled) with following the template below.

[dout_input] [x_input] [y_input] <- x_input/y_input is involved by compiling, because what tensor to use is determined when compiling time.

    |            |           |
   [ User Defined Backward Ops ]
                 |
      (values ∂out/∂x ∂out/∂y )
                 |
  [Move(x_in, ∂out/∂x)], [Move(y_in, ∂out/∂y)] (If the User-Defined-Backward Ops is ends with MoveTensorNode, this form is ignored.)

Lambda-Function:
Input : dout-past
Output: NIL

Inputs:

out-kernels ... nodes of user defined backward
inputs      ... inputs called with
"

  ;;FIX: REDUCE COMPILE TIME!
  
  ;; Collecting x_in

  (detach dout t)
  (let* ((inputs-in (loop for input in inputs-out
			  collect (detach (or (system-lazy-read-save-for-backward input) input) t)))
	 ;; Tracing User-Defined-Backward, still not yet compiled.
	 (out-kernels (apply #'backward node dout inputs-in))
	 (dout-place  (gensym "dout"))
	 ;; out-kernels = (list x.g y.g)
	 (out-kernels (loop with argn fixnum = (length inputs-in)
			    for x in out-kernels
			    for y in inputs-out
			    for i upfrom 0
			    collect (adjust-bw-place x y argn i))))
    
    (loop for kernel in out-kernels
	  collect
	  (when kernel
	    (let ((out (cl-waffe2/vm.generic-tensor:make-clone kernel)))
	      (cons
	       out
	       `(named-lambda ,(symb (class-name (class-of node)) '-backward) (,dout-place)
		  ;;(print "=======")
		  ;;(print ,dout)
		  (cl-waffe2/vm.generic-tensor:embody-actual-tensor
		   ,dout
		   ,dout-place)
		  ;;(print ,dout)
		  
		  ,(with-no-grad
		     (cl-waffe2/vm.generic-tensor:make-vm-function kernel)))))))))

(defun expand-backward-instant (node dout compile-option &rest inputs-out)
  (detach dout t)
  (let* ((inputs-in (loop for input in inputs-out
			  collect (detach (or (system-lazy-read-save-for-backward input) input) t)))
	 ;; Tracing User-Defined-Backward, still not yet compiled.
	 (out-kernels (apply #'backward node dout inputs-in))
	 ;; out-kernels = (list x.g y.g)
	 (out-kernels (loop with argn fixnum = (length inputs-in)
			    for x in out-kernels
			    for y in inputs-out
			    for i upfrom 0
			    collect (adjust-bw-place x y argn i))))
    (loop for kernel in out-kernels
	  collect
	  (when kernel
	    (list dout kernel compile-option inputs-out)))))

(defun call-instant-backward (outs)
  (multiple-value-bind (dout kernel compile-option inputs-out) (apply #'values outs)
    (with-no-grad
      (prog1
	  (cl-waffe2/vm.generic-tensor:run-node! kernel :compile-option compile-option)
	(detach dout nil)
	(map 'list #'(lambda (x) (detach x nil)) inputs-out)))))


;; the method backward constructs backward function
;; Constructing chains will be done at vm/generic-tensor/acceptor.lisp
(defmethod backward :around ((node AbstractNode) &rest inputs)
  (declare (ignore inputs))
  (when (not *no-grad*)
    (with-no-grad
      (with-shape-checkpoint (:backward node)
	(multiple-value-list (call-next-method))))))

(defmethod backward ((node AbstractNode) &rest inputs)
  (declare (ignore inputs))
  (error "Couldn't step backward because ~a backward is undefined." node))



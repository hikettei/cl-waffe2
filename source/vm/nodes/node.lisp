
(in-package :cl-waffe2/vm.nodes)

(defclass AbstractNode ()
  ((function-node
    :initarg
    :function-node
    :reader abstractnode-node
    :type function) ;; [x ~ y] [y z] -> [z x]
   (function-node1
    :initarg
    :function-node1
    :reader abstractnode-node1
    :type (or null function)) ;; [x y] [y z] -> [z x], ~ is removed. If ~ isn't used at function-node, set nil
   (transmission-state :initarg :transmission-state :reader transmission-state :type list)
   (ignore-shape-error :initform nil :accessor ignore-shape-error)
   (passed-at-least-once :initform nil :accessor node-passed-p :type boolean))
  (:documentation "The class AbstractNode is a fundamental object of describing computation nodes in cl-waffe.

AbstractNode must possess following:
   1. Transimission State
   2. Slots (for passing forward/backward)
   3. Variables (for building computation nodes)
"))

;; TODO: Under here.
(defmethod test-and-forward-shape ((node AbstractNode) &rest previous-shape)
  ""
  (funcall (abstractnode-node node) previous-shape))

(defun describe-problems (error-node detected-errors inputs outputs)
  ;; Enhancement:
  ;; Restart-Case
  ;; [Fix-Definition-And-Step]
  ;; [Replace-Shape-And-Step]
  ;; More Details:
  ;; Displays [pre-|post-]computation node
  ;; TODO: make it more intuitive....
  (shaping-error
   "Couldn't step forward because of shape-error.

At:     ~a
Inputs:          ~a
Excepted Output: ~a
Here's a list of reports.

1. ~a

~a
~a"
   error-node
   (map 'list #'shape inputs)
   outputs
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

;; Enhancement: A[~] -> B[~] <- replace A with input-name.
(defmethod forward :around ((node AbstractNode) &rest inputs)
  ;; Update Computation Nodes

  (let* ((transition-function     (abstractnode-node node))
	 (transition-function-sub (abstractnode-node1 node))
	 (pointer-states      (transmission-state node))
	 ;; What View To Use? modottara yaru.
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
	      (describe-problems node detected-errors inputs out-state)
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
						  :dtype (dtype (nth (or extend-from 0) inputs))
						  :order (order (nth (or extend-from 0) inputs))))
				    (state (make-statecontainer
					    :forward-out-form forward-form
					    :forward-n-out  (length out-state)
					    :backward-n-out (length input-states))))

			       ;; Move ga hikituganai...
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

(defmethod backward :around ((node AbstractNode) &rest inputs)
  (declare (ignore inputs))
  (when (not *no-grad*)
    (with-no-grad
      (multiple-value-list (call-next-method)))))

(defmethod backward ((node AbstractNode) &rest inputs)
  (error "Couldn't step backward because ~a backward is undefined." node))


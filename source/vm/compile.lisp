
(in-package :cl-waffe2/vm)

;; [TODO] Optimize compiling time, benchmark it!

;; Provides a compiler from cl-waffe2 IR -> Instruction Sequence

;; cl-waffe2 IR
;;     - Tree-structure
;;     - represented by AbstractTensor(S expression + Shape + CLOS Class)
;;     - or sometimes composable call-with-view

;; Instruction Sequence
;;    - Flatten
;;    - represetned by lambda expression

;; X <- Y
;; Y <- X optim: => Z <- Y
;; Z <- Y


(defparameter *vm-compile-option* :fastest)

(declaim (ftype (function (AbstractTensor) (or null WFInstruction)) ir->instruction))
(defun ir->instruction (tensor)
  "Reading a IR of tensor, the function returns a corresponding instruction"
  (declare (type AbstractTensor tensor))

  (cond
    ((null (tensor-backward tensor))
     ;; Has reached out the end of nodes.
     nil)
    (T
     (make-wfop
      (apply
       #'find-cached-function
       (statecontainer-forward-out-form (tensor-state tensor))
       (cl-waffe2/vm.generic-tensor::compile-option-form *vm-compile-option*)
       (tensor-variables tensor))
      tensor
      (tensor-backward tensor)
      (tensor-variables tensor)
      :out-to (node-out-to (tensor-backward tensor))
      :sv4bw  (node-sv4bw (tensor-backward tensor))))))

;;
;; Avoid duplicate compilation:
;;
;; (!sin x (!copy x))
;;

;; TODO:
;; Forward Reodering
;; MUL     VIEW
;; VIEW    VIEW
;; ADD  => MUL
;; VIEW    ADD  to compose more operations

(defun node-compile-into-vm (toplevel &key (fuse-p nil))
  "Set fuse-p=t to apply view-reordering and fuse operations"
  (declare (type AbstractTensor toplevel)
	   (optimize (speed 3)))
  (let ((instruction-seq)
	(variable-leaves)
	(seen nil))
    (declare (type list instruction-seq variable-leaves seen))
    ;; sorted-node
    ;; old
    ;; ...
    ;; new
    (loop with sorted-node = (topological-sort toplevel)
	  for tensor in sorted-node
	  for time fixnum upfrom 0
	  
	  if (null (tensor-backward tensor))
	    do (push tensor variable-leaves) ;; Register as a variable
	  else
	    do (when (not (detach-p tensor))

		 (when (null (find (tensor-iid tensor) seen :test #'eq))
		   (push (ir->instruction tensor) instruction-seq)
		   ;; Add to seen
		   (let ((bw (tensor-backward tensor)))
		     (when bw
		       (dolist (v (node-out-to bw))
			 (push (tensor-iid v) seen)))))

		 ;; Ask each devices if applying JIT?
		 (let ((result (cl-waffe2/vm.nodes::on-finalizing-compiling
				(tensor-backward tensor)
				tensor
				(nth (1+ time) sorted-node)
				nil)))
		   (when result
		     (push
		      ;; Embedding Lisp Code generated from JITxxxTensor.
		      (make-wfop
		       (compile
			nil
			`(lambda () ,result))
		       tensor
		       #'(lambda ()
			   ;; Displaying the information
			   (format nil "Foreign Function {~a}" (class-name (class-of tensor))))
		       nil)
		      instruction-seq)))))
    ;; Forward Mode ... (reverse instruction-seq)
    ;; Reverse Mode ... Trace tensors in instruction-seq order.
    
    ;; When no instructions is provided for toplevel
    ;; Just Return toplevel tensor.
    
    (when (null instruction-seq)
      (setq instruction-seq (list (%vm-wrap-tensor toplevel))))
    
    (if fuse-p
	(values
	 (reverse
	  (apply-path-fusion
	   (reverse instruction-seq)))
	 variable-leaves)
	(values instruction-seq variable-leaves))))

(defun forward->reverse-mode (iseq dout-toplevel &aux (dout-table (make-hash-table :test #'eq)))
  (declare (type list iseq)
	   (type AbstractTensor dout-toplevel))

  ;; WfInstruction: Op(Arg1 Arg2) -> Out1 Out2 ...
  ;; BW: g(dout, x1, x2, ...)

  (flet ((get-dout (tensor)
	   (gethash (tensor-iid tensor) dout-table))
	 (set-dout (tensor val)
	   (setf (gethash (tensor-iid tensor) dout-table) val)))

    (set-dout (wfop-self (car iseq)) dout-toplevel)
    
    (loop for inst of-type WfInstruction in iseq
	  if (get-dout (wfop-self inst))
	    append
	    (let* ((self   (wfop-self   inst))
		   (args   (wfop-args   inst)))
	      (append
	       ;; Backward_Kernel + Gradient_Adders (If Any)
	       (when (and (cl-waffe2/vm.generic-tensor::ancestor-param-p self)
			  (tensor-backward self))
		 (multiple-value-bind (bw-function node out-to directions) (make-backward-wfinst self)
		   (if (null bw-function)
		       nil
		       (progn
			 (loop for arg in args
			       for o   in out-to
			       for dir in directions
			       if dir
				 do (set-dout arg o)
				    (init-state-container! o))

			 (list
			  (make-wfop bw-function ;; ... dout var1 var2
				     self
				     node
				     `(,(get-dout self) ,@args)
				     :out-to (loop for o in out-to if o collect o)))))))
	       ;; Expand Gradient Adders
	       (loop for var in args
		     if (and (slot-value var 'requires-grad) (get-dout var))
		       append (expand-gradient-adder var (get-dout var))))))))


(defvar *compile-option* nil)
;; When doing forward: reverse it in advance
(defun compile-forward-and-backward (toplevel &key (need-backward t) (fuse-p t) (compile-mode :default))
  "

## [function] compile-forward-and-backward

```lisp
(compile-forward-and-backward toplevel &key (need-backward t) (fuse-p t) (compile-mode :default))
```

Compiles into cl-waffe2 IR from topleve to each leaf points (detach-p=t or backward=null variables). set `fuse-p`=t to get additional optimization to the generated IR.

Tips: `disassemble-waffe2-ir` to display compiled Instruction Sequence.

## Return

`(values forward-iseq backward-iseq leaves[an list of AbstractTensor that appeared in the node] dout)`
"
  (declare (type AbstractTensor toplevel))
  ;; fuse-p is intentionally disabled forcibly for a while
  ;; because it cause unexcepted behaviours
  (let ((*compile-option* (cl-waffe2/vm.generic-tensor::compile-option-form compile-mode)))
    (multiple-value-bind (iseq-forward leaves)
	(node-compile-into-vm toplevel :fuse-p fuse-p)

      (apply-in-place-mutation! iseq-forward leaves)

      (let* ((out-symbol-p (some #'symbolp (shape toplevel)))
	     (dout (when need-backward
		     (if (scalar-p toplevel)
			 (make-tensor 1 :dtype (dtype toplevel) :order (order toplevel))
			 (if out-symbol-p
			     (forward (cl-waffe2/base-impl:ScalarAdd (dtype toplevel))
				      (make-input (shape toplevel) nil
						  :dtype (dtype toplevel)
						  :order (order toplevel))
				      (make-tensor 1 :dtype (dtype toplevel)))
			     (make-tensor (shape toplevel) :initial-element 1 :dtype (dtype toplevel) :order (order toplevel))))))
	     (backward-iseq
	       (when (and need-backward
			  (ancestor-param-p toplevel))
		 (forward->reverse-mode iseq-forward dout))))

	;; Initializes Gradient Resetter
	(mapc
	 #'(lambda (tensor)
	     (when (slot-value tensor 'cl-waffe2/vm.generic-tensor:requires-grad)
	       (setf (cl-waffe2/vm.generic-tensor::gradient-resetter tensor)
		     (if (scalar-p tensor)
			 #'(lambda () (setf (tensor-vec (grad tensor)) (tensor-vec (make-tensor 0 :dtype (dtype tensor) :order (order tensor)))))
			 #'(lambda () (setf (tensor-grad-count tensor) 0))))))
	 leaves)

	(values (reverse iseq-forward)
		(if (and need-backward out-symbol-p (not (scalar-p toplevel)))
		    (append ;; dout = 0, so add 1
		     (reverse
		      (node-compile-into-vm dout))		     
		     backward-iseq)
		    backward-iseq)
		leaves
		dout)))))

(defun findout-origin (table tensor)
  (let ((last-ref (tensor-id tensor)))
    (loop while t do
      (if (null (gethash last-ref table))
	  (return-from findout-origin last-ref)
	  (setq last-ref (gethash last-ref table))))))

(defun disassemble-waffe2-ir (toplevel &key (backward t) (stream t) (fuse-p t))
  "
## [function] disassemble-waffe2-ir

```lisp
(disassemble-waffe2-ir toplevel &key (backward t) (stream t) (fuse-p t))
```

Prints out the compiled cl-waffe2 IR from toplevel to each leaf points to `stream`. If `backward` was set to t, `backward` is also displayed.
"
  (declare (type AbstractTensor toplevel))
  (multiple-value-bind (iseq-fw iseq-bw leaves)
      (compile-forward-and-backward toplevel :need-backward backward :fuse-p fuse-p)
    (declare (ignore leaves))
    
    (flet ((conc-iseq-str (iseq)
	     (let ((tensor-ids)
		   (scal-ids)
		   (tensor-table (make-hash-table)))
	       (with-output-to-string (out)
		 (with-indent-to iseq
		   (dolist (i iseq)
		     (when (and (movetensor-p (wfop-node i))
				(movetensor-ignore-me (wfop-node i)))
		       (setf (gethash (tensor-id (wfop-self i)) tensor-table) (tensor-id (second (wfop-args i)))))
		     (dolist (var (wfop-args i))
		       (if (scalar-p var)
			   (push (tensor-id var) scal-ids)
			   (push (findout-origin tensor-table var) tensor-ids)))
		     (princ i out)))
		 (format out "~%~a Instructions | ~a Tensors | ~a Scalars~%"
			 (length iseq)
			 (length (remove-duplicates tensor-ids))
			 (length (remove-duplicates scal-ids)))))))
      
      (format stream "~%disassemble-waffe2-ir:~% [Forward]: ~%~a~%" (conc-iseq-str iseq-fw))

      (format stream "~% [Pullback]: ~%~a~%" (conc-iseq-str iseq-bw))

      t)))

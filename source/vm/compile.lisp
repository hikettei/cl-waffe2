
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

(defun expand-gradient-adder (tensor grad)
  ;; Tensor += Grad
  (setf (detach-p grad) t)
  (prog1
      (let ((*no-grad* t))
	(reverse
	 (if (scalar-p tensor)
	     (progn
	       (node-compile-into-vm
		(forward
		 (cl-waffe2/base-impl::ScalarAndScalarAdd)
		 (grad tensor)
		 grad)))
	     (if (= (tensor-grad-count tensor) 0)
		 (progn
		   (node-compile-into-vm
		    (forward
		     (cl-waffe2/base-impl:MoveTensorNode (dtype tensor) :save-for-backward t)
		     (grad tensor)
		     grad)))
		 (progn
		   (node-compile-into-vm
		    (forward
		     (cl-waffe2/base-impl:AddNode (dtype tensor))
		     (grad tensor)
		     grad)))))))
    (setf (detach-p grad) nil)))

;; copy(sin(x, copy(x))) <- ???

(defun sv4bw-p (node)
  (and (movetensor-p node) ;; MoveTensor(SAVE_FOR_BACKWARD) isn't subject to backward. just move tensors
       (cl-waffe2/base-impl:mv-lazy-sv4bw node)))

(defun trace-backward-network (toplevel leaves dout-toplevel fuse-p)
  (declare (type AbstractTensor toplevel dout-toplevel))

  (sort-and-prune-for-backward toplevel dout-toplevel leaves fuse-p))

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

      ;; [TODO] Testing the line below carefully:
      ;; In-place mutation is working??
      (when (not fuse-p) ;; avoid twice-time applying
	(apply-in-place-mutation! iseq-forward leaves))

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
		 (trace-backward-network toplevel leaves dout fuse-p))))
	(mapc
	 #'(lambda (tensor)
	     (when (slot-value tensor 'cl-waffe2/vm.generic-tensor:requires-grad)
	       (setf (cl-waffe2/vm.generic-tensor::gradient-resetter tensor)
		     (if (scalar-p tensor)
			 #'(lambda () (setf (tensor-vec (grad tensor)) (tensor-vec (make-tensor 0 :dtype (dtype tensor) :order (order tensor)))))
			 #'(lambda () (setf (tensor-grad-count tensor) 0))))))
	
	;;		 (let* ((*no-grad* t)
	;;			(out (when (not (some #'symbolp (shape toplevel)))
	 ;;			       (build (cl-waffe2/base-impl:A*=scal (grad tensor) 0)))))
	 ;;		   (when (not (some #'symbolp (shape toplevel)))
	 ;;		     #'(lambda ()
	 ;;			 (forward out))))))))
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
	     (let ((tensor-ids))
	       (with-output-to-string (out)
		 (with-indent-to iseq
		   (dolist (i iseq)
		     (dolist (var (wfop-args i))
		       (push (tensor-id var) tensor-ids))
		     (princ i out)))
		 (format out "~%~a Instructions | ~a Tensors~%"
			 (length iseq)
			 (length (remove-duplicates tensor-ids)))))))

      (format stream "~%disassemble-waffe2-ir:~% [Forward ISeq]: ~%~a~%"  (conc-iseq-str iseq-fw))

      (format stream "~% [Backward ISeq]: ~%~a~%" (conc-iseq-str iseq-bw))

      t)))

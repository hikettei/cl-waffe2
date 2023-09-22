
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



(declaim (ftype (function (AbstractTensor) (or null WFInstruction)) ir->instruction))
(defun ir->instruction (tensor)
  "Reading a IR of tensor, the function returns a corresponding instruction"
  (declare (optimize (speed 3))
	   (type AbstractTensor tensor))

  (when (and (tensor-compiled-instruction-cache-fw tensor)
	     (equal (wfop-args (tensor-compiled-instruction-cache-fw tensor))
		    (tensor-variables tensor)))
    ;; Using The Cached Compiled Function Instead
    ;; At this time, variables shouldn't be changed
    (let ((out (tensor-compiled-instruction-cache-fw tensor)))
      (return-from ir->instruction out)))

  (cond
    ((null (tensor-backward tensor))
     ;; Has reached out the end of nodes.
     nil)
    (T
     (let ((result
	     (make-wfop
	      (apply
	       #'find-cached-function
	       (statecontainer-forward-out-form (tensor-state tensor))
	       *compile-option*
	       (tensor-variables tensor))
	      tensor
	      (tensor-backward tensor)
	      (tensor-variables tensor)
	      :out-to (node-out-to (tensor-backward tensor))
	      :sv4bw  (node-sv4bw (tensor-backward tensor)))))
       (setf (tensor-compiled-instruction-cache-fw tensor) result)
       result))))

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

		 (when (null (find (the symbol (tensor-iid tensor)) seen :test #'eql))
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

(defun forward->reverse-mode (iseq dout-toplevel &aux (dout-table (make-hash-table :test #'eql)))
  (declare (type list iseq)
	   (optimize (speed 3))
	   (type AbstractTensor dout-toplevel))

  ;; WfInstruction: Op(Arg1 Arg2) -> Out1 Out2 ...
  ;; BW: g(dout, x1, x2, ...)

  (flet ((get-dout (tensor)
	   (gethash (tensor-iid tensor) dout-table))
	 (set-dout (tensor val)
	   (setf (gethash (tensor-iid tensor) dout-table) val)))

    (set-dout (wfop-self (car iseq)) dout-toplevel)
    (loop ;;with grad-id-appeared-list list = nil
	  for inst of-type WfInstruction in iseq
	  if (get-dout (wfop-self inst))
	    append
	    (let* ((self   (wfop-self   inst))
		   (args   (wfop-args   inst)))
	      (append
	       ;; Backward_Kernel + Gradient_Adders (If Any)
	       (when (and (cl-waffe2/vm.generic-tensor::ancestor-param-p self)
			  (tensor-backward self)
			  (get-dout (wfop-self inst)))
		 (multiple-value-bind (bw-function node out-to directions bw-iseq) (make-backward-wfinst self (get-dout (wfop-self inst)))
		   (when bw-function
		     (loop for arg in args
			   for o   in out-to
			   for dir in directions
			   if dir
			     do (set-dout arg o)
				(init-state-container! o))

		     (let ((inst
			     (make-wfop bw-function ;; ... dout var1 var2
					self
					node
					`(,(get-dout (wfop-self inst)) ,@args)
					:out-to (loop for o in out-to if o collect o)
					:block-iseq bw-iseq)))	       
		       (list inst)))))
	       ;; Expand Gradient Adders
	       (loop for var in args
		     if (and (slot-value var 'requires-grad) (get-dout var))
		       append (let* ((grad (get-dout var))
				     (out  (expand-gradient-adder var grad)))
								  ;; Ensure that grad never conflicts
								  ;;:setq (not (find (the symbol (tensor-id grad)) grad-id-appeared-list)))))
				;;(push (tensor-id grad) grad-id-appeared-list)
				out)))))))


(defvar *compile-option* nil)
;; When doing forward: reverse it in advance
(defun compile-forward-and-backward (toplevel &key (need-backward t) (fuse-p t) (compile-mode :default) (optimize-locality t) (add1 t))
  "

## [function] compile-forward-and-backward

```lisp
(compile-forward-and-backward toplevel &key (need-backward t) (fuse-p t) (compile-mode :default) (optimize-locality t))
```

Compiles into cl-waffe2 IR (so-called iseq) from the given toplevel to each leaf points (where detach-p=t or backward=null variables). `toplevel` is AbstractTensor with backwards.

Tips: `disassemble-waffe2-ir` to display compiled Instruction Sequence.

### Return

`(values forward-iseq backward-iseq leaves[an list of AbstractTensor that appeared in the node] dout alloc-state)`
"
  (declare (type AbstractTensor toplevel))
  ;; fuse-p is intentionally disabled forcibly for a while
  ;; because it cause unexcepted behaviours
  (let ((*compile-option* (cl-waffe2/vm.generic-tensor::compile-option-form compile-mode)))
    (multiple-value-bind (iseq-forward leaves)
	(node-compile-into-vm toplevel :fuse-p fuse-p)
      ;; Set grad-count=0 if any
      (map 'list #'(lambda (tensor) (setf (tensor-grad-count tensor) 0)) leaves)

      (when optimize-locality
	;; Generating Setq{Pruned}
	(apply-in-place-mutation! iseq-forward leaves))

      (let* ((out-symbol-p (some #'symbolp (shape toplevel)))
	     (dout (when need-backward
		     (if (scalar-p toplevel)
			 (make-tensor 1 :dtype (dtype toplevel) :order (order toplevel))
			 (if out-symbol-p
			     (let ((dout-tensor (make-input (shape toplevel) nil
						  :dtype (dtype toplevel)
						  :order (order toplevel))))
			       (if add1
				   (forward (cl-waffe2/base-impl:ScalarAdd (dtype toplevel))
					    dout-tensor					    
					    (make-tensor 1 :dtype (dtype toplevel)))
				   dout-tensor))
			     (make-tensor (shape toplevel) :initial-element 1 :dtype (dtype toplevel) :order (order toplevel))))))
	     (backward-iseq
	       (when (and need-backward
			  (ancestor-param-p toplevel))
		 (when optimize-locality
		   (setf (tensor-protect-me dout) t))
		 (forward->reverse-mode iseq-forward dout))))

	(when optimize-locality
	  (setq iseq-forward (eliminate-setq-node iseq-forward)))
	
	(let ((forward  (reverse iseq-forward))
	      (backward (if (and need-backward out-symbol-p (not (scalar-p toplevel)))
			    (append
			     (reverse
			      (node-compile-into-vm dout))
			     backward-iseq)
			    backward-iseq)))
	  
	  (multiple-value-bind (bw allocation) (when optimize-locality (optimize-memory-locality! forward backward))
	    (values forward (or bw backward) leaves dout allocation)))))))

#+sbcl(setf sb-ext:*inline-expansion-limit* 4)
(defun findout-origin (table tensor &key (limit 10))
  (declare (type hash-table table)
	   (type AbstractTensor tensor)
	   (optimize (speed 3))
	   (type fixnum limit)
	   #+sbcl(inline findout-origin))
  (let ((last-ref (tensor-id tensor)))
    (loop while t for n fixnum upfrom 0 do
      (if (> n limit) (return-from findout-origin last-ref))      
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
		   (scal-ids))
	       (with-output-to-string (out)
		 (with-indent-to iseq
		   (dolist (i iseq)
		     (dolist (var (wfop-args i))
		       (if (scalar-p var)
			   (push (tensor-id var) scal-ids)
			   (push (tensor-id var) tensor-ids)))
		     (princ i out)))
		 (format out "~%~a Instructions | ~a Tensors | ~a Scalars~%"
			 (length iseq)
			 (length (remove-duplicates tensor-ids))
			 (length (remove-duplicates scal-ids)))))))
      
      (format stream "~%disassemble-waffe2-ir:~% [Forward]: ~%~a~%" (conc-iseq-str iseq-fw))
      (format stream "~% [Pullback]: ~%~a~%" (conc-iseq-str iseq-bw))
      t)))

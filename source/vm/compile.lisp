
(in-package :cl-waffe2/vm)


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


(defparameter *vm-compile-option* :default)

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
      (tensor-variables tensor)))))

;;
;; Avoid duplicate compilation:
;;
;; (!sin x (!copy x))
;;

(defun node-compile-into-vm (toplevel)
  (let ((instruction-seq)
	(variable-leaves))

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
		 (push (ir->instruction tensor) instruction-seq)

		 ;; Ask each devices if applying JIT?
		 (let ((result (cl-waffe2/vm.nodes::on-finalizing-compiling
				(tensor-backward tensor)
				tensor
				(nth (1+ time) sorted-node)
				nil)))
		   (when result
		     (push
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
    (values instruction-seq variable-leaves)))

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
	     (progn
	       (node-compile-into-vm
		(forward
		 (cl-waffe2/base-impl:AddNode (dtype tensor))
		 (grad tensor)
		 grad))))))
    (setf (detach-p grad) nil)))

;; copy(sin(x, copy(x))) <- ???

(defun sv4bw-p (node)
  (and (movetensor-p node) ;; MoveTensor(SAVE_FOR_BACKWARD) isn't subject to backward. just move tensors
       (cl-waffe2/base-impl:mv-lazy-sv4bw node)))


(defun trace-backward-network (toplevel leaves dout-toplevel)
  (declare (type AbstractTensor toplevel dout-toplevel))

  (sort-and-prune-for-backward toplevel dout-toplevel leaves))

;; When doing forward: reverse it in advance
(defun compile-forward-and-backward (toplevel &key (need-backward t))
  "Compiles into cl-waffe2 IR from topleve to each leaf points (detach-p=t or backward=null variables).

`disassemble-waffe2-ir` to display compiled Instruction Sequence."
  (declare (type AbstractTensor toplevel))
  (multiple-value-bind (iseq-forward leaves)
      (node-compile-into-vm toplevel)

    ;; [TODO] Testing the line below carefully:
    ;; In-place mutation is working??
    (apply-in-place-mutation! iseq-forward leaves)

    (let* ((dout (if (scalar-p toplevel)
		     (make-tensor 1 :dtype (dtype toplevel) :order (order toplevel))
		     (make-tensor (shape toplevel) :initial-element 1 :dtype (dtype toplevel) :order (order toplevel))))
	   (backward-iseq
	     (when (and need-backward
			(ancestor-param-p toplevel))
	       (trace-backward-network toplevel leaves dout))))

      (mapc
       #'(lambda (tensor)
	   (when (slot-value tensor 'cl-waffe2/vm.generic-tensor:requires-grad)
	     (setf (cl-waffe2/vm.generic-tensor::gradient-resetter tensor)
		   (if (scalar-p tensor)
		       #'(lambda () (setf (tensor-vec (grad tensor)) (tensor-vec (make-tensor 0 :dtype (dtype tensor) :order (order tensor)))))
		       (let* ((*no-grad* t)
			      (out (build (cl-waffe2/base-impl:A*=scal (grad tensor) 0))))
			 #'(lambda ()
			     (forward out)))))))
       leaves)
      ;; (print (reverse iseq-forward))
      ;; (print backward-iseq)
      ;; (print backward-iseq)

      (values (reverse iseq-forward) backward-iseq leaves))))


(defun disassemble-waffe2-ir (toplevel &key (backward t) (stream t))
  "
## [function] disassemble-waffe2-ir

Prints out the compiled cl-waffe2 IR from toplevel to each leaf points to `stream`. If `backward` was set to t, `backward` is also displayed."
  (declare (type AbstractTensor toplevel))
  (multiple-value-bind (iseq-fw iseq-bw leaves)
      (compile-forward-and-backward toplevel :need-backward backward)
    (declare (ignore leaves))
    
    (flet ((conc-iseq-str (iseq)
	     (with-output-to-string (out)
	       (dolist (i iseq)
		 (princ i out)))))

      (format stream "~%== [disassemble-waffe2-ir: Forward] ======~%~a~%" (conc-iseq-str iseq-fw))

      (format stream "~%== [disassemble-waffe2-ir: Backward] ======~%~a~%" (conc-iseq-str iseq-bw))

      t)))

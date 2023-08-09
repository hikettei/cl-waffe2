
(in-package :cl-waffe2/vm)


;; Provides a compiler from cl-waffe2 IR -> Instruction Sequence

;; cl-waffe2 IR
;;     - Tree-structure
;;     - represented by AbstractTensor(S expression + Shape + CLOS Class)
;;     - or sometimes composable call-with-view

;; Instruction Sequence
;;    - Flatten
;;    - represetned by lambda expression

;; TODO: 後で全て強制的にbuildを用いるモードでテストしないといけない
;; Github Actionsではパラメーターを変えて二回テストしよう
;; JITCPUTensorの仕様は後で変更しよう...
;; TODO:
;; Optimize:

;; X <- Y
;; Y <- X optim: => Z <- Y
;; Z <- Y


(defparameter *vm-compile-option* :default)

(declaim (ftype (function (AbstractTensor) (or null WFInstruction)) ir->instruction))
(defun ir->instruction (tensor)
  "Reading a IR of tensor, the function returns a corresponding instruction"
  (declare (type AbstractTensor tensor))

  ;;(reset-compiled-function-cache!)
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
    (loop with sorted-node = (topological-sort toplevel)
	  for tensor in sorted-node
	  if (null (tensor-backward tensor))
	    do (push tensor variable-leaves) ;; Register as a variable
	  else
	    do (push (ir->instruction tensor) instruction-seq))
    ;; Forward Mode ... (reverse instruction-seq)
    ;; Reverse Mode ... Trace tensors in instruction-seq order.
    (values instruction-seq variable-leaves)))

(defun expand-gradient-adder (tensor grad)
  ;; Tensor += Grad
  (setf (detach-p grad) t)
  (prog1
      (let ((*no-grad* t))
	(map
	 'list
	 #'(lambda (x)
	     (setf (wfop-bw-is-leaf-p x) t)
	     x)
	 (list
	  (car
	   (if (scalar-p tensor)
	       (node-compile-into-vm
		(forward
		 (cl-waffe2/base-impl::ScalarAndScalarAdd)
		 (grad tensor)
		 grad))
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

(defun trace-backward-network (toplevel leaves dout-toplevel)
  (declare (type AbstractTensor toplevel dout-toplevel))

  (let ((sorted-fw-tensors (topological-sort-in-backward-direction toplevel dout-toplevel))
	(compiled-code))

    (loop for fw-instruction of-type list in sorted-fw-tensors do
      (let* ((prev-dout   (car fw-instruction))
	     (fw-tensor   (second fw-instruction))
	     (backward-kernels
	       (apply
		#'cl-waffe2/vm.nodes:compiler-expand-backward
		(tensor-backward fw-tensor)
		prev-dout
		(tensor-variables fw-tensor))))
	(loop for grad in backward-kernels
	      for var  in (tensor-variables fw-tensor)
	      if (and grad (ancestor-param-p var))
		do (setq compiled-code `(,@compiled-code ,@(reverse (node-compile-into-vm grad))))
	      if (and grad (null (tensor-backward var)) (slot-value var 'cl-waffe2/vm.generic-tensor:requires-grad))
		do (setq compiled-code `(,@compiled-code ,@(expand-gradient-adder var grad))))))
    compiled-code))

;; When doing forward: reverse it in advance
(defun compile-forward-and-backward (toplevel &key
						(need-backward t))
  ""
  (declare (type AbstractTensor toplevel))
  (multiple-value-bind (iseq-forward leaves)
      (node-compile-into-vm toplevel)

    (print iseq-forward)
    (apply-in-place-mutation! iseq-forward leaves)

    (let* ((dout (if (scalar-p toplevel)
		   (make-tensor 1 :dtype (dtype toplevel) :order (order toplevel))
		   (make-tensor (shape toplevel) :initial-element 1 :dtype (dtype toplevel) :order (order toplevel))))
	   (backward-iseq
	     (when need-backward
	       (trace-backward-network toplevel leaves dout))))

      (print backward-iseq)
      (values (reverse iseq-forward) backward-iseq leaves))))



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

;; TODO: Optimizer Compiling time?

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

(defun trace-backward-network (instruction-seq leaves dout-toplevel)
  (declare (type list instruction-seq leaves)
	   (type AbstractTensor dout-toplevel))

  (let ((set-of-backward-node)
	(grad-table (make-hash-table)))

    ;; Starting from dout-toplevel
    (setf (gethash (tensor-id (wfop-self (car instruction-seq))) grad-table) dout-toplevel)

    ;; set-of-backward-node ...
    ;; an list of
    ;; (!cos (!copy ..))
    ;; (!sin (!copy ...)) <- each of them are the definition of backward.

    ;; ISeq (Forward mode)
    ;; Inst [ADD-CPU] X <- X Y    ,@(BACKWARD Inst [ADD-CPU] X -> X Y)
    ;; Inst [SUB-CPU] Z <- K X => ,@(BACKWARD Inst [ADD-CPU] X -> X Y)
    ;;         ..
    
    (loop for inst of-type WFInstruction in instruction-seq do
      (let ((backward-kernels
	      (apply
	       #'cl-waffe2/vm.nodes:compiler-expand-backward
	       (tensor-backward (wfop-self inst))
	       (gethash (tensor-id (wfop-self inst)) grad-table)
	       (wfop-args inst))))

	(loop for var in (wfop-args inst)
	      for grad in backward-kernels
	      if grad
		do (setf (gethash (tensor-id var) grad-table) grad)
	      if (and grad
		      (ancestor-param-p (wfop-self inst))
		      ;;(ancestor-param-p var)
		      )
		do (setq set-of-backward-node
			 `(,@set-of-backward-node
			   ,@(reverse (node-compile-into-vm grad)))))

	;; Accumulate gradients when reatched the end of nodes.

	(loop for var in (wfop-args inst)
	      for grad in backward-kernels do
	  ;; The next node does not exist?
	  (loop for maybe-param in (tensor-variables var)
		if (and (null (tensor-backward maybe-param))
			(null (tensor-variables maybe-param))
			grad
			(slot-value maybe-param 'cl-waffe2/vm.generic-tensor::requires-grad))
		  ;; Accumulate the gradients
		  do (setq set-of-backward-node
			   `(,@set-of-backward-node
			     ;; Expand Gradient Adder
			     ,@(expand-gradient-adder
				maybe-param
				(gethash (tensor-id var) grad-table))))))))
    ;; Eliminate Duplicated Node
    ;; [Created Backward Iseq]
    ;; t
    ;; 0 X <- X + Y
    ;; 1 Y <- X + Y
    ;; 2   ...
    ;; ...

    ;; TP Sort -> In-place mutation -> VM
    (multiple-value-bind (backward-iseq adders) (topological-sort-iseq set-of-backward-node)
      (let ((backward-iseq `(,@backward-iseq ,@adders)))
	(apply-in-place-mutation! backward-iseq leaves)
	backward-iseq))))

;; When doing forward: reverse it in advance
(defun compile-forward-and-backward (toplevel &key
						(need-backward t))
  ""
  (declare (type AbstractTensor toplevel))
  (multiple-value-bind (iseq-forward leaves)
      (node-compile-into-vm toplevel)

    (apply-in-place-mutation! iseq-forward leaves)

    (let ((backward-iseq
	    (when need-backward
	      (trace-backward-network
	       iseq-forward
	       leaves
	       (if (scalar-p toplevel)
		   (make-tensor 1 :dtype (dtype toplevel) :order (order toplevel))
		   (make-tensor (shape toplevel) :initial-element 1 :dtype (dtype toplevel) :order (order toplevel)))))))
      
      (values (reverse iseq-forward) backward-iseq leaves))))


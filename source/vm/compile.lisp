
(in-package :cl-waffe2/vm)


;; Provides a compiler from cl-waffe2 IR -> Instruction Sequence

;; cl-waffe2 IR
;;     - Tree-structure
;;     - represented by AbstractTensor(S expression + Shape + CLOS Class)
;;     - or sometimes composable call-with-view

;; Instruction Sequence
;;    - Flatten
;;    - represetned by lambda expression

;; JITCPUTensorの仕様は後で変更しよう...

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
    (loop with sorted-node = (topological-sort toplevel)
	  for tensor in sorted-node
	  if (null (tensor-backward tensor))
	    do (push tensor variable-leaves) ;; Register as a variable
	  else
	    do (push (ir->instruction tensor) instruction-seq))
    ;; Forward Mode ... (reverse instruction-seq)
    ;; Reverse Mode ... Trace tensors in instruction-seq order.
    (values instruction-seq variable-leaves)))

(defun trace-backward-network (instruction-seq leaves dout-toplevel)
  (declare (type list instruction-seq leaves)
	   (type AbstractTensor dout-toplevel))

  (let ((set-of-backward-node)
	(prev-dout dout-toplevel))

    ;; set-of-backward-node ...
    ;; an list of
    ;; (!cos (!copy ..))
    ;; (!sin (!copy ...)) <- each of them are the definition of backward.

    ;; forwardモードの計算ノードの順番にInstructionSeqを辿る
    
    (loop for inst of-type WFInstruction in instruction-seq
	  do (push
	      (cl-waffe2/vm.nodes:expand-backward
	       (tensor-backward (wfop-self inst))
	      ))))

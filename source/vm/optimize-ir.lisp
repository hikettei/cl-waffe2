
(in-package :cl-waffe2/vm)

(defun make-loadp (node-name from-tensor to-tensor)
  "Reshape/Permute/View etc ... -> Loadp(pointer)
from-tensor* = to-tensor*
[op=Loadp]"

  (make-wfop
   #'(lambda (from to)
       (setf (tensor-vec from) (tensor-vec to))
       from)
   from-tensor
   #'(lambda ()
       (format nil "Loadp(from: ~a)" node-name))
   `(,from-tensor ,to-tensor)
   :out-to `(,from-tensor)
   :loadp t))

;; Reshape/LazyTranspose Moving/Replacing
;; Permute/View loadp mutation
(defun simplify-iseq (iseq)
  "Returns a list of instructions applied these operations:
- Moving ReshapeTensor to the top of instructions, places Loadp where it was
- Deletes an instruction which is a subclass of Rebundant-Node class.
- Sets loadp=t if the node is a subclass of Loadp-Node"
  ;; Subtype of LoadpInstruction class, SystemIR class (lazy-cons)
  ;; Reshape ... 一番上に持ってくる + Replace with make-loadp
  ;; Rebundant-Node
  ;; Reshapeじゃなくて...Loadp-But-KeepMe-Node
  ;; Replace-Loadp-Node

  (loop for inst in iseq
	if (and
	    (not (typep (wfop-node inst) 'cl-waffe2/base-impl:Rebundant-Node)))
	  collect
	  (typecase (wfop-node inst)
	    (cl-waffe2/base-impl:load-myself-node
	     (make-loadp
	      (class-name (class-of (wfop-node inst)))
	      (wfop-self inst)
	      (car (wfop-args inst))))
	    (T
	     inst))))



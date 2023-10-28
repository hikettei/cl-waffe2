
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

(defun read-loadp (instruction)
  "Return: (values A B) where A* = B* if loadp=t, otherwise returns nil"
  (when (wfop-loadp instruction)
    (if (typep (wfop-node instruction) 'cl-waffe2/base-impl:Loadp-Node-Rev)
	(values
	 (wfop-self instruction)
	 (first  (wfop-args instruction)))
	(values
	 (wfop-self instruction)
	 (second (wfop-args instruction))))))

;; Reshape/LazyTranspose Moving/Replacing
;; Permute/View loadp mutation
(defun simplify-iseq (iseq)
  "Returns a list of instructions applied these operations:
- Moving ReshapeTensor to the top of instructions, places Loadp where it was
- Deletes an instruction which is a subclass of Rebundant-Node class.
- Sets loadp=t if the node is a subclass of Loadp-Node"

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
	    (cl-waffe2/base-impl:loadp-node
	     (assert
	      (cl-waffe2/base-impl:ensure-loadp (wfop-node inst) (wfop-args inst) (wfop-out-to inst))
	      ()
	      "simplify-iseq: Can't simplify this node: ~a
because the transformation wasn't described in A <= A, B forms." inst)
	     (setf (wfop-loadp inst) t)
	     inst)
	    (cl-waffe2/base-impl:loadp-node-rev
	     (setf (wfop-loadp inst) t)
	     inst)
	    (T
	     inst))))



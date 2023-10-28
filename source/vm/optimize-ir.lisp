
(in-package :cl-waffe2/vm)

(defun make-loadp (node-name from-tensor to-tensor)
  "Reshape/Permute/View etc ... -> Loadp(pointer)
to-tensor* = from-tensor*

[op=Loadp]"

  (make-wfop
   #'(lambda (from to)
       (setf (tensor-vec from) (tensor-vec to))
       from)
   to-tensor
   #'(lambda ()
       (format nil "Loadp(from: ~a)" node-name))
   `(,from-tensor ,to-tensor)
   :out-to `(,to-tensor)
   :loadp t))

;;(print (make-loadp 'aaa (make-tensor 1) (make-tensor 1)))

(defun reduce-ir-diversity (iseq)
  ""
  ;; Subtype of LoadpInstruction class, SystemIR class (lazy-cons)
  ;; Reshape ... 一番上に持ってくる + Replace with make-loadp
  )

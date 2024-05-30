
(in-package :cl-waffe2/backends.aten)

(defclass AtenOp ()
  ((blueprint :accessor aten-bp :type string)
   (composite :accessor aten-composite :type aten/ir:composite)
   (code      :accessor aten-compiled-code :type string)
   (inputs    :accessor aten-inputs)
   (outputs   :accessor aten-outputs)))

;; Aten[Clang]
(defclass Aten (AbstractTensor) nil
  (:documentation "
## [AbstractTensor] Aten
Base class for various aten backends.
"))

;; define-symbol-macro suru
(defclass Aten[Clang] (Aten cl-waffe2/backends.lisp:LispTensor)
  nil)

(defclass Aten[Metal] (Aten cl-waffe2/backends.lisp:LispTensor)
  nil)

;; [MEMO] How to implement cuda backend?
;; (defclass Aten[CUDA])

;; [TO ADD] Cast, Ax+BNode
;; [TODO] Remove ./JITCPUTensor

(defmethod ->composite ((op AtenOp))
  (aten/ir::make-composite
   :inputs (remove-duplicates (map 'list (alexandria:compose #'aten/ir:parse-aten #'tensor->shape-tracker) (aten-inputs op)) :test #'equal)
   :outputs (map 'list #'(lambda (x) (format nil "~a" (tensor-id x))) (aten-outputs op))
   :name (format nil "~a" (gensym "K"))
   :code (aten-bp op)))

(defun compile-composite (composite)
  (let* ((uops (aten/lang:Trace-uops
		(aten/ir:composite-inputs composite)
		(read-from-string (aten/ir:composite-code composite))))
	 (graph (aten/engine:uops-optimize uops)))
    (multiple-value-bind (cmp code) (aten/engine:realize graph composite)
      (values cmp code))))

(defmethod %compile ((self AtenOp))
  (multiple-value-bind (cmp code) (compile-composite (->composite self))
    (format t "Compiling: ~a~%" self)
    (print code)
    (setf (aten-composite self) cmp
	  (aten-compiled-code self) code)))

(defmethod on-finalizing-compiling ((device-name (eql 'Aten)) iseq-fw iseq-bw)
  (clang::set-clang-runtime);; tmp
  
  (aten/engine:initialize-runtime (aten/engine::runtimeconfig-name aten/engine::*runtime*) nil)
  (flet ((process (x)
	   (dolist (wfir x)
	     (when (typep (wf/vm:wfop-node wfir) 'AtenOp)
	       (%compile (wf/vm:wfop-node wfir))
	       (setf (wf/vm:wfop-op wfir)
		     #'(lambda (&rest args)
			 (apply (aten/engine::cc-caller (aten-composite (wf/vm:wfop-node wfir))) (map 'list #'tensor-vec args))			 
			 (apply #'values (wf/vm:wfop-out-to wfir))))))))
    (process iseq-fw)
    (process iseq-bw))
  (values iseq-fw iseq-bw))


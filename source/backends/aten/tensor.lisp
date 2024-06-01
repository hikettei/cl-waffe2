
(in-package :cl-waffe2/backends.aten)

(defclass AtenOp ()
  ((blueprint :accessor aten-bp :type Blueprint)
   (scalars   :accessor aten-scalars :type hash-table)
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

(defun rest->alist (rest)
  (loop for i upfrom 0 below (length rest) by 2
	if (not (find (nth i rest) `(:debug)))
	  collect (cons (symbol-name (nth i rest)) (nth (1+ i) rest))))

(macrolet ((define-aten-backend (name initializer &rest slots)
	     `(progn	       
		(define-symbol-macro ,name `(,',name Aten))
		(defmacro ,name (&rest configs &key (debug 0) &allow-other-keys)
		  "(with-devices ((Aten[Clang] :debug 3)) ...)"
		  `(progn
		     (,',initializer)
		     (aten/engine:initialize-runtime
		      (aten/engine::runtimeconfig-name aten/engine::*runtime*)
		      (rest->alist ',configs))
		     (setf (aten/engine::runtimeconfig-debug aten/engine::*runtime*) ,debug)
		     ,',name))		
		(defclass ,name (Aten cl-waffe2/backends.lisp:LispTensor) ,slots))))
  (define-aten-backend
      Aten[Clang]
    clang::set-clang-runtime))


;; [TO ADD] Cast, Ax+BNode
;; [TODO] Remove ./JITCPUTensor
(defun idkey (x) (format nil "~(~a~)" x))

(defmethod ->composite ((op AtenOp))
  (let ((scalars (loop with declared = (apply #'append (map 'list #'shape (aten-inputs op)))
		       for name in (bp-deps (aten-bp op))
		       if (null (find name declared :test #'equal))
			 collect name)))

    (let ((id2table (make-hash-table :test #'equal)))
      (loop for name in (bp-deps (aten-bp op))
	    do (setf (gethash (idkey name) id2table) name))
      (setf (aten-scalars op) id2table))
    (aten/ir::make-composite
     :inputs (remove-duplicates
	      (append
	       (map
		'list
		(alexandria:compose #'aten/ir:parse-aten #'tensor->shape-tracker)
		(aten-inputs op))
	       (loop for name in scalars
		     collect (aten/ir::%parse-aten (coerce (format nil "~a{Int}[]<>()" name) '(simple-array character (*))))))
	      :test #'equal)
     :outputs (map 'list #'(lambda (x) (format nil "~a" (tensor-id x))) (aten-outputs op))
     :name (format nil "~a" (gensym "K"))
     :code (bp-code (aten-bp op)))))

(defun composite-inputs (cc)
  (let ((header (aten/engine::cc-defun-header cc)))
    (loop for arg in (aten/engine::uop-defun-inputs header)
	  for scalar-p = (null (aten/ir:aten-shape arg))
	  for type     = (aten/ir:aten-type-class arg)
	  for shape    = (aten/ir:aten-shape arg)
	  if scalar-p collect arg)))

(defun compile-composite (composite)
  (let* ((uops (aten/lang:trace-uops
		(aten/ir:composite-inputs composite)
		(read-from-string (aten/ir:composite-code composite))))
	 (graph (aten/engine:uops-optimize uops)))
    (multiple-value-bind (cmp code) (aten/engine:realize graph composite)
      (values cmp code))))

(defmethod %compile ((self AtenOp))
  (aten/engine::with-debug-level (1)
    (format t "[Aten] Compiling: ~a~%" self))
  (multiple-value-bind (cmp code) (compile-composite (->composite self))
    (setf (aten-composite self) cmp
	  (aten-compiled-code self) code)))

(defmethod on-finalizing-compiling ((device-name (eql 'Aten)) iseq-fw iseq-bw)
  (when (null aten/engine::*runtime*)
    (error "aten/engine::*runtime* is not declared.~%First, initialize the runtime using (Aten[Backend] ...) macro."))

  (setf (aten/engine::runtimeconfig-indexing-rule aten/engine::*runtime*) :ndarray
        (aten/engine::runtimeconfig-vectorize-strategy aten/engine::*runtime*) :disabled)
  (flet ((process (x)
	   (dolist (wfir x)
	     (when (typep (wf/vm:wfop-node wfir) 'AtenOp)
	       (%compile (wf/vm:wfop-node wfir))
	       (let* ((inputs   (map 'list #'aten/ir:aten-id (composite-inputs (aten-composite (wf/vm:wfop-node wfir)))))
		      (id2table (aten-scalars (wf/vm:wfop-node wfir)))
		      (inputs   (map 'list #'(lambda (x) (gethash (idkey x) id2table)) inputs)))
		 (setf (wf/vm:wfop-op wfir)
		       #'(lambda (&rest args)
			   (apply
			    (aten/engine::cc-caller (aten-composite (wf/vm:wfop-node wfir)))
			    (append
			     (map 'list #'tensor-vec args)
			     (map 'list #'cl-waffe2/vm::maybe-observe-axis inputs)))
			   (apply #'values (wf/vm:wfop-out-to wfir)))))))))
    (process iseq-fw)
    (process iseq-bw))
  (values iseq-fw iseq-bw))



(in-package :cl-waffe2.frontends/onnx)

;; ~~ Converter ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(defparameter *converter-features* (make-hash-table :test #'equal))
(defmacro defop ((opset-name min-opset-version) ((gph inputs attrs &rest more) &body body))
  "Defines a converter"
  (let ((tmp (gensym)))
    `(let ((,tmp #'(lambda (,gph ,inputs ,attrs ,@more) (declare (ignorable ,gph)) ,@body)))
       (if (gethash ,opset-name *converter-features*)
	   (let ((values (gethash ,opset-name *converter-features*)))
	     (let ((position (position ,min-opset-version values :test #'= :key #'cdr)))
	       (if position
		   (setf (nth position (gethash ,opset-name *converter-features*)) (cons ,tmp ,min-opset-version))
		   (push (cons ,tmp ,min-opset-version) (gethash ,opset-name *converter-features*)))))
	   (setf (gethash ,opset-name *converter-features*) (list (cons ,tmp ,min-opset-version)))))))

(defun get-converter (op-type opset-version)
  (declare (type string op-type)
	   (type fixnum opset-version))

  (let ((candidates (gethash op-type *converter-features*)))
    (when (null candidates)
      (error "get-converter: Convertion pattern for ~a is not defined yet." op-type))

    (let ((candidates (sort candidates #'> :key #'cdr)))
      (loop for (impl . version) in candidates
	    if (<= version opset-version)
	      do (return-from get-converter impl)))
    (error "get-converter: there's no implementation for ~a satisfying opset=~a.~%Candidates:~a"
	   op-type
	   opset-version
	   candidates)))
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defstruct (Graph-Proto-Helper
	    (:conc-name gp-)
	    (:constructor make-graph-proto-helper (graph-proto opset-version)))
  (graph-proto grpah-proto :type Graph-Proto)
  (opset-version opset-version :type fixnum)
  ;; ID -> Waffe2 Tensor
  (name2value (make-hash-table :test #'equal) :type hash-table))

(defun tensor-shape-proto.dimension->aten (proto)
  (declare (type tensor-shape-proto.dimension proto))
  (let ((value (oneof-value (tensor-shape-proto.dimension-value proto))))
    (etypecase value
      (number value)
      (string (intern value)))))

(defun tensor-proto->aten (tensor-proto)
  (declare (type tensor-proto tensor-proto))
  (let ((array (raw->array tensor-proto)))
    ;; [TODO] Add change-facet function when adding a support for CUDA.
    (change-facet array :direction 'AbstractTensor)))

(defun value-info-proto->aten (value-info-proto)
  (declare (type Value-Info-Proto value-info-proto))
  (with-slots ((name cl-onnx::name) (type cl-onnx::type)) value-info-proto
    (let ((type (oneof-value (type-proto-value type))))
      (typecase type
	(type-proto.tensor
	 (with-slots ((elem-type cl-onnx::elem-type) (shape cl-onnx::shape)) type
	   (let* ((dims (tensor-shape-proto-dim shape))
		  (dims (map 'list #'tensor-shape-proto.dimension->aten dims)))
	     (make-input dims (intern name "KEYWORD") :dtype (int->data-type elem-type)))))
	(T
	 (error "[from-onnx] value-info-proto->aten: Not implemented: ~a" type))))))

(defun value->users (graph-proto name)
  (declare (type string name)
	   (type graph-proto graph-proto))
  (loop for node in (graph-proto-node graph-proto)
	for position upfrom 0
	if (find name (node-proto-input node) :test #'equal)
	  collect node))

(defun user->values (graph-proto name)
  (declare (type string name)
	   (type graph-proto graph-proto))
  (loop for node in (graph-proto-node graph-proto)
	for position upfrom 0
	if (find name (node-proto-output node) :test #'equal)
	  collect node))

(defun gp-name->value (graph-proto-helper name)
  (declare (type string name))
  (gethash name (gp-name2value graph-proto-helper)))

(defun (setf gp-name->value) (value graph-proto-helper name)
  (setf (gethash name (gp-name2value graph-proto-helper)) value))

(defun node-proto->aten (graph-proto-helper node-proto)
  (declare (type graph-proto-helper graph-proto-helper)
	   (type node-proto node-proto))
  
  (dolist (input (node-proto-input node-proto))
    (when (null (gethash input (gp-name2value graph-proto-helper)))
      (let ((values (user->values (gp-graph-proto graph-proto-helper) input)))
	(dolist (val values)
	  (node-proto->aten graph-proto-helper val)))))
  
  (let* ((converter (get-converter (node-proto-op-type node-proto) (gp-opset-version graph-proto-helper)))
	 (input
	   (map
	    'list
	    #'(lambda (x)
		(let ((value (gp-name->value graph-proto-helper x)))
		  (when (null value) (error "node-proto->aten: ~a is not declared?" x))
		  value))
	    (node-proto-input node-proto)))
	 (attrs (make-hash-table :test #'equal)))

    (dolist (attr (node-proto-attribute node-proto))
      (setf (gethash (attribute-proto-name attr) attrs)
	    (cl-onnx::read-attr attr)))

    (let ((out-in-tensors
	    (multiple-value-list (funcall converter graph-proto-helper input attrs))) ;; attrs params
	  (output
	    (node-proto-output node-proto)))
      
      (assert (= (length output) (length out-in-tensors))
	      ()
	      "Assertion Failed: Converter for (~a opset=~a) expected to return ~a tensors butgot ~a."
	      (node-proto-op-type node-proto)
	      (gp-opset-version graph-proto-helper)
	      (length output)
	      (length out-in-tensors))

      (loop for tensor in out-in-tensors
	    for name   in output
	    do (setf (gp-name->value graph-proto-helper name) tensor))

      (apply #'values out-in-tensors))))

(defun from-onnx (graph-proto-helper)
  "A helper function to translate graph-proto"
  (declare (type Graph-Proto-Helper graph-proto-helper))

  
  (let* ((graph-proto (gp-graph-proto graph-proto-helper))
	 (input-names (map 'list #'value-info-proto-name (graph-proto-input graph-proto)))
	 (output-names (map 'list #'value-info-proto-name (graph-proto-output graph-proto)))
	 (inputs (map 'list #'value-info-proto->aten (graph-proto-input graph-proto))))

    (loop for name in input-names
	  for input in inputs
	  do (setf (gp-name->value graph-proto-helper name) input))
    
    (dolist (initializer (graph-proto-initializer graph-proto))
      ;; Initializer = tensor-proto
      (setf (gp-name->value graph-proto-helper (tensor-proto-name initializer))
	    (tensor-proto->aten initializer)))

    (assert (null (graph-proto-sparse-initializer graph-proto)) () "[TODO] not ready")

    (let ((innermosts
	    (apply #'append (map 'list #'(lambda (x) (user->values graph-proto x)) output-names))))
      (dolist (node-proto innermosts)
	(node-proto->aten graph-proto-helper node-proto)))

    (let ((results (map 'list #'(lambda (x) (gp-name->value graph-proto-helper x)) output-names)))

      (when (some #'null results)
	(error "Failed to trace the graph: ~a -> ~a" output-names results))

      (if (<= (length results) 1)
	  (car results)
	  (apply #'cl-waffe2/base-impl:lazy-values results)))))


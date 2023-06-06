
(in-package :cl-waffe2/backends.cpu)

(defclass CPUTensor (AbstractTensor) nil)

(defmethod initialize-instance :before ((tensor CPUTensor)
					&rest initargs
					&key &allow-other-keys)
  ;; if projected-p -> alloc new vec
  (let* ((shape (getf initargs :shape))
	(dtype (dtype->lisp-type (getf initargs :dtype)))
	(vec   (getf initargs :vec))
	(facet (getf initargs :facet))
	(initial-element (coerce (or (getf initargs :initial-element) 0) dtype)))
    (when (eql facet :exist)
      (if vec
	  (setf (tensor-vec tensor) vec)
	  (setf (tensor-vec tensor)
		(make-array
		 (apply #'* shape)
		 :element-type dtype
		 :initial-element initial-element))))))


(defun tensor-ptr (tensor &key (offset 0))
  (declare (type CPUTensor tensor)
	   (type fixnum offset)
	   (optimize (speed 3) (safety 0)))
  #+sbcl
  (let ((ptr (sb-sys:vector-sap (sb-ext:array-storage-vector (the (simple-array * (*)) (tensor-vec tensor))))))
    (incf-pointer ptr (* (the fixnum (foreign-type-size (dtype tensor))) offset)))
  #-(or sbcl)
  (error "CPUTensor requires SBCL!"))




(in-package :cl-waffe2/backends.lisp)

(defclass LispTensor (AbstractTensor) nil
  (:documentation "
## [AbstractTensor] LispTensor
"))

(defmethod current-backend-state ((backend-name (eql 'LispTensor)))
  "Portable Common Lisp implementation of matrix operations")

(defmethod initialize-instance :before ((tensor LispTensor)
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


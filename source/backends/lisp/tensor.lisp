
(in-package :cl-waffe2/backends.lisp)

(defclass LispTensor (AbstractTensor) nil)

(defmethod initialize-instance :before ((tensor LispTensor)
					&rest initargs
					&key &allow-other-keys)
  ;; if projected-p -> alloc new vec
  (let ((shape (getf initargs :shape))
	(dtype (dtype->lisp-type (getf initargs :dtype)))
	(vec   (getf initargs :vec))
	(facet (getf initargs :facet)))
    (when (eql facet :exist)
      (if vec
	  (setf (tensor-vec tensor) vec)
	  (setf (tensor-vec tensor)
		(make-array
		 (apply #'* shape)
		 :element-type dtype))))))


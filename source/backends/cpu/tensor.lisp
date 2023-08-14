
(in-package :cl-waffe2/backends.cpu)

(defclass CPUTensor (AbstractTensor) nil
  (:documentation "
## [AbstractTensor] CPUTensor

"))

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

;; could be optimized...
(declaim (inline tensor-ptr))
(defun tensor-ptr (tensor &key (offset 0))
  (declare (type CPUTensor tensor)
	   (type fixnum offset))
  #+sbcl
  (let ((ptr (sb-sys:vector-sap (sb-ext:array-storage-vector (the (simple-array * (*)) (tensor-vec tensor))))))
    (locally (declare (optimize (speed 1) (safety 0)))
      (incf-pointer ptr (the fixnum (* (the fixnum (foreign-type-size (dtype tensor))) offset)))))
  #-(or sbcl)
  (error "CPUTensor requires SBCL!"))

(declaim ;;(inline incf-tensor-ptr)
	 (ftype (function (AbstractTensor cffi-sys:foreign-pointer &key (:offset fixnum)) cffi-sys:foreign-pointer) incf-tensor-ptr))
(defun incf-tensor-ptr (tensor ptr &key (offset 0))
  (declare (type AbstractTensor tensor)
	   (type cffi-sys:foreign-pointer ptr)
	   (type fixnum offset))
  (let ((out (the fixnum (* (the fixnum (foreign-type-size (dtype tensor))) offset))))
    (if (= out 0)
	ptr
        (progn
	  (cffi:incf-pointer ptr out)))))


(defmacro with-tensor-ptr ((bind tensor) &body body)
  `(progn
     ;; Ensure that tensor storage vector has allocated.
     (tensor-vec ,tensor)
     (cffi:with-pointer-to-vector-data (,bind (cl-waffe2/vm.generic-tensor::vec ,tensor))
       (declare (type cffi-sys:foreign-pointer ,bind))
       ,@body)))

(defmacro with-tensor-ptrs ((&rest input-forms) &body body)
  (labels ((expand (rest-forms)
	     (if rest-forms
		 `(with-tensor-ptr (,(caar rest-forms) ,(second (car rest-forms)))
		    ,(expand (cdr rest-forms)))
		 `(progn ,@body))))
    (expand input-forms)))


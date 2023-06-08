
(in-package :cl-waffe2)

(defmacro with-build ((forward variables parameters) out &body body)
  `(multiple-value-bind (,forward ,variables, parameters) (construct-forward ,out)
     ,@body))

(defmacro with-cpu (&body body)
  "TODO: Docstring"
  #+sbcl
  `(with-devices (CPUTensor LispTensor)
     ,@body)
  #-sbcl
  `(with-devices (LispTensor)
     ,@body))

;; (defmacro with-cuda)

(defmacro with-dtype (dtype &body body)
  `(let ((*default-dtype* ,dtype))
     ,@body))

(defmacro with-column-major (&body body)
  `(let ((*default-order* :column))
     ,@body))

(defmacro with-row-major (&body body)
  `(let ((*default-order* :row))
     ,@body))

(defmacro with-config ((&key
			  ;; TO ADD:
			  ;; Use-Dtype
			  ;; Matmul-Accuracy
			  ;; 
			  (device :cpu)
			  (no-grad nil)
			  (dtype :float)
			  (order :column))
		       &body
			 body)
  "Integrates all the annoying configs."
  (declare (type (and keyword (member :cpu :cuda)) device)
	   (type (and keyword (member :row :column)) order)
	   (type keyword dtype))
  `(,(case device
       (:cpu 'with-cpu)
       (:cuda 'with-cuda))
    (,(if no-grad
	  'with-no-grad
	  'progn)
     (,(case order
	 (:column 'with-column-major)
	 (:row    'with-row-major))
      (with-dtype ,dtype
	,@body)))))


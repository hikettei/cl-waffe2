
(in-package :cl-waffe2)

;; Here's a utility macro which configurates vm/node's setting.

;; TODO: Place here, with-devices
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

;; Broadcast_Auto shouldn't be modular, all the nodes defined in cl-waffe2, should work under all combines of config.

(defmacro with-config ((&key
			  ;; TO ADD:
			  ;; Global Dtype
			  ;; (sin uint8) -> Global Float Dtype
			  ;; Matmul-Accuracy
			  ;; 
			  (device :cpu)
			  (no-grad nil)
			  (dtype :float)
			  (order :column)
			  (num-cores 1))
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
	(with-num-cores (,num-cores)
	  ,@body))))))


;; TODO: Add set-config for REPL.



(defun collect-initarg-slots (slots constructor-arguments)
  (map 'list #'(lambda (slots)
		 ;; Auto-Generated Constructor is Enabled Only When:
		 ;; slot has :initarg
		 ;; slot-name corresponds with any of constructor-arguments
		 (when
		     (and
		      (find (first slots) (alexandria:flatten constructor-arguments))
		      (find :initarg slots))
		   slots))
       slots))


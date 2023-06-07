
(in-package :cl-waffe2)

(defmacro with-build ((forward variables parameters) out &body body)
  `(multiple-value-bind (,forward ,variables, parameters) (construct-forward ,out)
     ,@body))

;; TODO
;; (defmacro with-config :DTYPE, DEVICE, etc..)


(defmacro with-cpu (&body body)
  "TODO: Docstring"
  #+sbcl
  `(with-devices (CPUTensor LispTensor)
     ,@body)
  #-sbcl
  `(with-devices (LispTensor)
     ,@body))

;; (defmacro with-cuda


;;Add: numcl matmul
(defun lin (weight x bias)
  (!add (!matmul weight x) (!view bias t `(:broadcast ,(second (shape x))))))

;; !matmul with uniform-random...? (Can't cast)
(defun test-lin ()
  (with-cpu
    (let ((x      (make-tensor `(100 100)))
	  (weight (make-tensor `(100 100) :requires-grad t))
	  (bias   (make-tensor `(100 1)   :requires-grad t)))
      (multiple-value-bind (fw bw vars pms) (build (!sum (lin weight x bias)))
	(time (funcall fw))
	(time (funcall bw))
	(grad weight)
	))))

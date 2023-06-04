
(in-package :cl-waffe2/base-impl)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defclass TMPDevice (AbstractTensor) nil))

(defmacro with-tmp-device (&body body)
  `(let ((*using-backend* `(,@*using-backend* TMPDevice))
	 (*facet-monopoly-mode* NIL))
     ,@body))


(eval-when (:compile-toplevel :load-toplevel :execute)
  (defnode (MoveTensorNode (myself)
	    :where `(A[~] B[~] -> A[~])
	    :slots ((ignore-me :initform nil :accessor movetensor-ignore-me :type boolean)
		    (save-for-backward :initform nil :accessor movetensor-save-for-backward :type boolean)) ;; when t, ignored.
	    :documentation "
The Node MoveTensorNode must satisfy the following behaviours:

Forward:
1. If ignore-me is t, return the given value itself.
2. Otherwise, move x <- y.

Note that until (tensor-vec) is called, x is never allocated.

The option ignore-me can be accessed by the function (movetensor-ignore-me MoveTensorNode)")))


(defun !move (place tensor)
  "TODO: DOCSTRING"
  (forward (MoveTensorNode) place tensor))

(defun !copy (tensor)
  "TODO: DOCSTRING"
  (let ((out (make-input (shape tensor) nil
			 :dtype (dtype tensor)
			 :order (order tensor))))
    (!move out tensor)))


(defnode (ViewTensorNode (myself result before)
	  :where `(A[result] B[before] -> A[result]
			    where before = ',before result = ',result))
  (setf (ignore-shape-error myself) t))

(define-impl (ViewTensorNode :device TMPDevice)
	     :forward
	     ((self viewed-tensor old)
	      (declare (ignore old))
	      `(progn ,viewed-tensor))
	     :backward
	     ((self dout dx dy)
	      (declare (ignore dout))
	      (values dx dy)))

(defun !view (tensor &rest subscripts)
  "TODO: DOC"
  (let ((out (apply #'cl-waffe2/vm.generic-tensor::view tensor subscripts)))
    ;; Update Chains
    (with-tmp-device
      (let ((result (forward (ViewTensorNode (shape out) (shape tensor)) (detach out) tensor)))
	(setf (cl-waffe2/vm.generic-tensor:ancestor-param-p out)
	      (cl-waffe2/vm.generic-tensor:ancestor-param-p result)
	      
	      (tensor-out-n out) (tensor-out-n result)
	      (tensor-state out) (tensor-state result)
	      
	      (tensor-backward out) (tensor-backward result)  
	      (tensor-variables out) (tensor-variables result))
	out))))


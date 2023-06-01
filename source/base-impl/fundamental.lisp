
(in-package :cl-waffe2/base-impl)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defclass TMPDevice (AbstractTensor) nil))

(defmacro with-tmp-device (&body body)
  `(let ((*using-backend* `(,@*using-backend* TMPDevice))
	 (*facet-monopoly-mode* NIL))
     ,@body))


(eval-when (:compile-toplevel :load-toplevel :execute)
  (defnode (MoveTensorNode (myself)
	    :where `([~] [~] -> [~])
	    :slots ((ignore-me :initform nil :accessor movetensor-ignore-me :type boolean)
		    (save-for-backward :initform nil :accessor movetensor-save-for-backward :type boolean)) ;; when t, ignored.
	    :documentation "
The Node MoveTensorNode must satisfy the following behaviours:

Forward:
1. If ignore-me is t, return the given value itself.
2. Otherwise, move x <- y.

Note that until (tensor-vec) is called, x is never allocated.

The option ignore-me can be accessed by the function (movetensor-ignore-me MoveTensorNode)")))

(defmacro with-export (name &body body)
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (export ',name)
     ,@body))


(defun !move (place tensor)
  "TODO: DOCSTRING"
  (forward (MoveTensorNode) place tensor))

(defun !copy (tensor)
  "TODO: DOCSTRING"
  (let ((out (make-input (shape tensor) nil
			 :dtype (dtype tensor)
			 :order (order tensor))))
    (!move out tensor)))

(defnode (ViewTensorNode (myself result)
	  :where `([~ result] [~] -> [result] where result = ',result)))

(define-impl (ViewTensorNode :device TMPDevice)
	     :forward
	     ((self viewed-tensor old)
	      (declare (ignore old))
	      `(progn ,viewed-tensor))
	     :backward
	     ((self dout dx dy)
	      (declare (ignore dx dy))
	      (values dout dout)))

(defun !view (tensor &rest subscripts)
  (let ((out (apply #'cl-waffe2/vm.generic-tensor::view tensor subscripts)))
    ;; Update Chains
    (with-tmp-device
      (forward (ViewTensorNode (shape out)) out tensor))))


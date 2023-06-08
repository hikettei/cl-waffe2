
(in-package :cl-waffe2/base-impl)

(defnode (MoveTensorNode (myself)
	  :where `(A[~] B[~] -> A[~])
	  :slots ((ignore-me :initform nil :accessor movetensor-ignore-me :type boolean)
		  (save-for-backward :initform nil :accessor movetensor-save-for-backward :type boolean)) ;; when t, ignored.
	  :backward ((self dout dx dy)
		     (declare (ignore dx))
		     (let ((dy-out
			     (if (and
				  (eql (tensor-attribute dy) :chain)
				  (movetensor-ignore-me self))
				 dout
				 (!copy dout))))
		       ;; side eff
		       (values dout dy-out)))
	  :documentation "
The Node MoveTensorNode must satisfy the following behaviours:

Forward:
1. If ignore-me is t, return the given value itself.
2. Otherwise, move x <- y.

Note that until (tensor-vec) is called, x is never allocated.

The option ignore-me can be accessed by the function (movetensor-ignore-me MoveTensorNode)"))


(defun !move (place tensor)
  "TODO: DOCSTRING"
  (forward (MoveTensorNode) place tensor))

(defun !copy (tensor)
  "TODO: DOCSTRING"
  (let ((out (make-input (shape tensor) nil
			 :dtype (dtype tensor)
			 :order (order tensor))))
    (!move out tensor)))

(defnode (ViewTensorNode (myself subscripts result before)
	  :slots ((subscripts :initarg :subscripts))
	  :where `(A[result] B[before] -> A[result]
			     where before = ',before result = ',result))
  (setf (ignore-shape-error myself) t))

(define-impl (ViewTensorNode)
	     :forward
	     ((self viewed-tensor old)
	      (declare (ignore old))
	      `(progn
		 ,viewed-tensor))
	     :backward
	     ((self dout dx dy)
	      (let ((out-sub (tensor-view dy))
		    (inp-sub (slot-value self 'subscripts)))
		(values
		 nil
		 (!move
		  dy
		  (apply
		   #'!view
		   (!move dx (apply #'!view dout inp-sub))
		   out-sub))))))

(defun !view (tensor &rest subscripts)
  "TODO: DOC"
  (let ((out (apply #'cl-waffe2/vm.generic-tensor::view tensor subscripts)))
    ;; Update Chains
    (forward (ViewTensorNode subscripts (shape out) (shape tensor)) out tensor)))


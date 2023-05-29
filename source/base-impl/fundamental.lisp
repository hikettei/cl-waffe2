
(in-package :cl-waffe2/base-impl)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defclass TMPDevice (AbstractTensor) nil))

(defnode (ViewNode (myself tensor &rest subscripts)
	  :where `([~ before-shape] -> [after-shape]
				  where
				  before-shape = ',(shape tensor)
				  after-shape  = ',(compute-visible-shape (shape tensor) subscripts))
	  :slots ((view)
		  (old-shape))
	  :documentation "")
  (setf (slot-value myself 'view) subscripts)
  (setf (slot-value myself 'old-shape) (shape tensor)))

(define-impl (ViewNode :device TMPDevice)
	     :forward ((self x)
		       `(values
			 ,(apply
			   #'make-view
			   x
			   (slot-value self 'view))))
	     :backward ((self dy)
			;; TODO: backward
			`(values ,dy)))

(defmacro with-tmp-device (&body body)
  `(let ((*using-backend* `(,@*using-backend* TMPDevice))
	 (*facet-monopoly-mode* NIL))
     ,@body))

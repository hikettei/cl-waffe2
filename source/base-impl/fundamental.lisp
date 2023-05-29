
(in-package :cl-waffe2/base-impl)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defclass TMPDevice (AbstractTensor) nil))


(defmacro with-tmp-device (&body body)
  `(let ((*using-backend* `(,@*using-backend* TMPDevice))
	 (*facet-monopoly-mode* NIL))
     ,@body))


(eval-when (:compile-toplevel :load-toplevel :execute)
  (export 'MoveTensorNode)
  (defnode (MoveTensorNode (myself)
	    :where `([~] [~] -> [~])
	    :slots ((ignore-me :initarg nil)) ;; when t, ignored.
	    :documentation "")))

(defmacro with-export (name &body body)
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (export ',name)
     ,@body))


(with-export !move
  (defun !move (place tensor)
    "TODO: DOCSTRING"
    (forward (MoveTensorNode) place tensor)))


(with-export !copy
  (defun !copy (tensor)
    "TODO: DOCSTRING"
    (let ((out (make-input (shape tensor) nil
			   :dtype (dtype tensor)
			   :order (order tensor))))
      (!move out tensor))))


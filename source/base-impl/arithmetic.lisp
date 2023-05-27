
(in-package :cl-waffe2/base-impl)

(macrolet ((define-arithmetic-node (name document save-for-backward)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defnode (,name (myself)
			  :where `([~] [~] -> [~])
			  :slots ,(if save-for-backward
				      `((out :initarg :out :accessor node-out)
					(x :accessor node-x)
					(y :accessor node-y))
				      `((out :initarg :out :accessor node-out)))
			  :documentation ,(format nil "[Node for Arithmetic Operation]: ~a

(node-out node) ... the tensor to store result.
(node-x node) (node-y node) ... the tensor copied to compute backward." document))))))
  (define-arithmetic-node AddNode "Computes x + y element-wise." nil)
  (define-arithmetic-node SubNode "Computes x - y element-wise." nil)
  (define-arithmetic-node MulNode "Computes x * y element-wise." nil)
  (define-arithmetic-node DivNode "Computes x / y element-wise." nil))


;; 「!」 key can be hit in both the JP and EN sequences without breaking the home position.

;; TODO: Document
;; TODO: Automatically Dispatch: scalar-add etc... depending or their types.
(macrolet ((define-arithmetic-node-caller (name node-name document)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y)
		  ,document
		  (forward (,node-name) x y)))))
  (define-arithmetic-node-caller
      !add
    AddNode
    "x <- x + y")
  (define-arithmetic-node-caller
      !sub
    SubNode
    "x <- x - y")
  (define-arithmetic-node-caller
      !mul
    MulNode
    "x <- x * y")
  (define-arithmetic-node-caller
      !div
    DivNode
    "x <- x / y"))


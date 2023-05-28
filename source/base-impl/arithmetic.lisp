
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

;; TODO: backward
(defun !sum (tensor &key (axis t) (out nil))
    "Sum up the tensor along axis.
axis = nil or fixnum or list."
  (declare (type AbstractTensor tensor)
	   (type (or t list fixnum) axis))
  (let* ((shape (copy-list (shape tensor)))
	 (view-args (make-list (length shape) :initial-element t))
	 (dims  (length shape)))

    ;; Compute Reduction Size.
    (typecase axis
      (fixnum
       (if (< axis 0)
	   (setq axis (+ axis dims)))

       (setf (nth axis view-args) `(:broadcast ,(nth axis shape)))
       (setf (nth axis shape) 1))
      (list
       (dolist (dim axis)
	 (let ((tgt (if (< dim 0)
			(+ dim dims)
			dim)))
	   (setf (nth tgt view-args) `(:broadcast ,(nth tgt shape)))
	   (setf (nth tgt shape) 1))))
      (T
       (setq view-args (loop for s in shape
			     collect `(:broadcast ,s)))
       (setq shape (make-list dims :initial-element 1))))

    (let ((out (or out (make-tensor
			shape
			:dtype (dtype tensor)
			:order (slot-value tensor 'cl-waffe2/vm.generic-tensor::order)))))

      (assert (equal (shape out) shape)
	      nil
	      "!sum: Assertion Failed because the given out's shape is ~a, but excepted: ~a" (shape out) shape)

      (let ((out* (apply #'view out view-args)))
	(!add out* tensor)))))




(in-package :cl-waffe2/vm.generic-tensor)

(defclass DebugTensor (AbstractTensor) nil) ;; ANSI CL's (make-array)

(defclass ScalarTensor (AbstractTensor)
  nil
  (:documentation "The class ScalarTensor, is used to represent scalar-object."))

(defmethod initialize-instance :before ((tensor ScalarTensor) &rest initargs &key &allow-other-keys)

  (let ((vec (getf initargs :vec)))
    (assert (typep vec 'number) nil
	    "Assertion Failed with (typep vec 'number). but got: ~a" vec))

  (setf (slot-value tensor 'orig-shape)    `(1)
	(slot-value tensor 'visible-shape) `(1)))


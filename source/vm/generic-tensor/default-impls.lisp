
(in-package :cl-waffe2/vm.generic-tensor)

(defclass ScalarTensor (AbstractTensor)
  nil
  (:documentation "The class ScalarTensor, is used to represent scalar values."))

(defmethod initialize-instance :before ((tensor ScalarTensor) &rest initargs &key &allow-other-keys)
  (declare (ignore initargs))

  ;; ScalarTensor can include symbol.
  (setf (slot-value tensor 'orig-shape)    `(1)
	(slot-value tensor 'visible-shape) `(1)))

(defmethod vref ((tensor ScalarTensor) index)
  (declare (ignore index))
  (tensor-vec tensor))

(defmethod current-backend-state ((backend-name (eql 'ScalarTensor)))
  "is a special tensor for representing scalar values.")


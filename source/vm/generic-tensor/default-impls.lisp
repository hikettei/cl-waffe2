
(in-package :cl-waffe2/vm.generic-tensor)

(defclass DebugTensor (AbstractTensor) nil)

(defclass ScalarTensor (AbstractTensor)
  nil
  (:documentation "The class ScalarTensor, is used to represent scalar-object."))

(defmethod initialize-instance :before ((tensor ScalarTensor) &rest initargs &key &allow-other-keys)
  (declare (ignore initargs))

  (setf (slot-value tensor 'orig-shape)    `(1)
	(slot-value tensor 'visible-shape) `(1)))

(defmethod vref ((tensor ScalarTensor) index)
  (declare (ignore index))
  (tensor-vec tensor))


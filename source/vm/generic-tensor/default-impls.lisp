
(in-package :cl-waffe2/vm.generic-tensor)

(defclass DebugTensor (AbstractTensor) nil) ;; ANSI CL's (make-array)

(defclass ScalarTensor (AbstractTensor)
  nil
  (:documentation "The class ScalarTensor, is used to represent scalar-object."))

(defmethod initialize-instance :before ((tensor ScalarTensor) &rest initargs &key &allow-other-keys)

  ;; ErrorCheck Here.
  (declare (ignore initargs))

  )

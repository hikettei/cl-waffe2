
(in-package :cl-waffe2/backends.jit.lisp)

(defclass JITLispTensor (cl-waffe2/backends.lisp:LispTensor) nil
  (:documentation "
## [AbstractTensor] JITLispTensor
"))

(defmethod current-backend-state ((backend-name (eql 'JITLispTensor)))
  "To be deleted in the future release. do not use this.")


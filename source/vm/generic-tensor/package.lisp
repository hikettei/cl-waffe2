
(in-package :cl-user)

(defpackage :cl-waffe2/vm.generic-tensor
  (:use :cl)

  ;; Tensor classes
  (:export
   #:AbstractTensor
   #:CPUTensor

   #:tensor-backward
   #:tensor-variables
   #:tensor-state
   #:tensor-out-n
   )

  ;; Conditions
  (:export
   #:shaping-error)

  ;; APIs for StateContainer
  (:export
   #:statecontainer
   #:make-statecontainer
   )

  ;; Backends / Tensor API
  (:export
   #:shape
   #:make-tensor
   #:*using-backend*))

(in-package :cl-waffe2/vm.generic-tensor)


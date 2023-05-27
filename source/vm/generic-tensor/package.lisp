
(in-package :cl-user)

(defpackage :cl-waffe2/vm.generic-tensor
  (:use :cl)

  ;; Tensor classes
  (:export
   #:AbstractTensor

   #:tensor-backward
   #:tensor-variables
   #:tensor-state
   #:tensor-out-n
   #:tensor-vec
   #:tensor-facet
   #:tensor-name
   )

  ;; Dtype API
  (:export
   #:dtype-t
   #:dtype->lisp-type
   )

  ;; Conditions
  (:export
   #:shaping-error)

  (:export
   #:viewinstruction
   #:viewinstruction-offset
   #:viewinstruction-size
   #:call-with-view)
  ;; APIs for StateContainer
  (:export
   #:statecontainer
   #:make-statecontainer
   )

  (:export
   #:embody-tensor
   #:construct-forward)

  ;; Backends / Tensor API
  (:export
   #:shape
   #:make-input
   #:make-tensor
   #:*using-backend*))

(in-package :cl-waffe2/vm.generic-tensor)


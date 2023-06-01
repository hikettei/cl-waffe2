
(in-package :cl-user)

(defpackage :cl-waffe2/vm.generic-tensor
  (:use :cl :lparallel)

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
   #:dtype
   #:mref
   #:vref
   #:grad
   #:order
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
   #:*no-grad*
   #:with-no-grad)

  (:export
   #:viewinstruction
   #:viewinstruction-offset
   #:viewinstruction-size
   #:viewinstruction-by
   #:call-with-view
   #:compute-visible-shape)
  ;; APIs for StateContainer
  (:export
   #:statecontainer
   #:make-statecontainer
   )

  (:export
   #:shape-equal)

  (:export
   #:embody-input
   #:build
   #:construct-forward
   #:construct-backward
   )

  ;; Backends / Tensor API
  (:export
   #:shape
   #:make-input
   #:make-tensor
   #:*using-backend*))

(in-package :cl-waffe2/vm.generic-tensor)


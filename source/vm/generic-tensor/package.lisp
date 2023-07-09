
(in-package :cl-user)

(defpackage :cl-waffe2/vm.generic-tensor
  (:use :cl :lparallel)

  (:export
   ;;#:*cache-directory*
   #:with-memory-pool
   #:make-clone
   #:print-current-memory-pool
   #:free-current-memory-pool
   #:make-compiled-kernel
   #:compile-forward-kernel
   #:*memory-pool*)
  ;; Tensor classes
  (:export
   #:AbstractTensor
   #:ScalarTensor

   #:tensor-backward
   #:tensor-variables
   #:tensor-state
   #:tensor-out-n
   #:tensor-vec
   #:tensor-facet
   #:tensor-stride
   #:tensor-name
   #:dtype
   #:tensor-attribute
   #:tensor-protect-me
   #:requires-grad
   #:ancestor-param-p
   #:actual-shape
   #:mref
   #:vref
   #:grad
   #:order
   #:scalar-p
   #:detach-p
   #:tensor-projected-p
   #:tensor-flexible-p
   #:view
   #:tensor-view
   #:viewtype
   #:detach!
   #:*default-dtype*
   #:*default-order*
   #:set-save-for-backward
   #:read-save-for-backward
   #:parameter
   #:*with-printing-tensor-omitted*
   #:tensor-id

   #:compile-option-t
   )

  (:export
   #:shaping-error)

  ;; Dtype API
  (:export
   #:dtype-t
   #:dtype->lisp-type
   #:dtype-of
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
   #:stride-of
   #:size-of
   #:offset-of
   #:compute-visible-shape
   #:force-list)
  ;; APIs for StateContainer
  (:export
   #:statecontainer
   #:make-statecontainer
   )

  (:export
   #:make-vm-function
   #:compile-forward-chain)
  
  (:export
   #:movetensor-p
   #:shape-equal)

  (:export
   #:embody-input
   #:embody-actual-tensor
   #:compiled-composite
   #:build
   #:set-input
   #:get-input)

  ;; Backends / Tensor API
  (:export
   #:shape
   #:total
   #:dims
   #:make-input
   #:make-tensor
   #:*using-backend*))

(in-package :cl-waffe2/vm.generic-tensor)


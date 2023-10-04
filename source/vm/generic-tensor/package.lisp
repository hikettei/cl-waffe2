
(in-package :cl-user)

(defpackage :cl-waffe2/vm.generic-tensor
  (:use :cl :lparallel :bordeaux-threads :cl-waffe2/threads)

  (:export
   #:State-Dict
   #:State-dict-table
   #:make-state-dict
   #:from-state-dict
   #:parse-state-dict-key

   #:load-state-dict)
  
  (:export
   ;;#:*cache-directory*
   #:with-memory-pool
   #:make-clone
   #:print-current-memory-pool
   #:free-current-memory-pool
   #:make-compiled-kernel
   #:find-cached-function
   #:*memory-pool*
   #:*static-alloc-state*)
  ;; Tensor classes

  (:export
   #:AbstractLoop
   #:aloop-rank
   #:aloop-size
   #:aloop-mode
   #:aloop-element-n
   #:aloop-by)
  
  (:export
   #:AbstractTensor
   #:ScalarTensor

   #:current-backend-state
   #:tensor-compiled-instruction-cache-fw
   #:tensor-compiled-instruction-cache-bw
   #:tensor-id-lock-p
   #:tensor-finalizer
   #:tensor-grad-count
   #:tensor-backward
   #:read-result
   #:tensor-variables
   #:tensor-state
   #:tensor-out-n
   #:tensor-vec
   #:tensor-facet
   #:tensor-actual-stride
   #:tensor-state-dict-name
   #:tensor-param-belongs-to
   #:tensor-stride
   #:tensor-name
   #:shape-with-broadcastable
   #:tensor-initial-offset
   #:dtype
   #:tensor-attribute
   #:tensor-protect-me
   #:requires-grad
   #:ancestor-param-p
   #:original-shape
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
   #:parameter
   #:*with-printing-tensor-omitted*
   #:tensor-id
   #:tensor-iid
   #:system-lazy-set-save-for-backward
   #:system-lazy-read-save-for-backward
   #:compile-option-t
   )

  (:export
   #:hook-optimizer!
   #:call-optimizer!
   #:reset-grad!)

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
   #:with-ranked-loop
   
   #:do-compiled-loop
   ;;#:do-compiled-loop*
   #:solve-loop-order
   
   #:stride-of
   #:size-of
   #:offset-of
   #:compute-visible-shape
   #:force-list
   #:permute*)
  ;; APIs for StateContainer
  (:export
   #:statecontainer
   #:statecontainer-forward-out-form
   #:make-statecontainer

   #:compiled-kernel-body
   )

  (:export
   #:make-vm-function)
  
  (:export
   #:movetensor-p
   #:shape-equal)

  (:export
   #:embody-input
   #:embody-actual-tensor
   #:compiled-composite
   #:compiled-variables
   #:nodevariables-parameters
   #:build
   #:reset-compiled-function-cache!
   #:model-parameters
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

(defparameter *static-alloc-state* nil)



(in-package :cl-user)

(defpackage :cl-waffe2/vm
  (:use
   :cl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl)
  (:export
   ;; LazyAxis
   #:LazyAxis
   #:lazyaxis-arguments
   #:lazyaxis-constraints
   #:lazyaxis-id
   #:lazyaxis-symbol
   #:symbol-lazyaxis
   #:make-lazyaxis
   #:observe-axis
   #:maybe-observe-axis
   #:make-lazy-assert
   #:make-higher-order-lazyaxis

   #:with-fixing-adjustable-shape
   #:tensor-fix-adjustable-shape)
  (:export
   #:*opt-level*
   #:*logging-vm-execution*   
   #:accept-instructions
   #:compile-forward-and-backward
   #:disassemble-waffe2-ir
   #:benchmark-accept-instructions)
  (:export
   #:WfInstruction
   #:wfop-op
   #:wfop-self
   #:wfop-out-to
   #:wfop-node
   #:wfop-sv4bw
   #:wfop-args
   #:wfop-loadp))

(in-package :cl-waffe2/vm)

;; Provides cl-waffe2 VM, IR, ... without running (compile nil ..)


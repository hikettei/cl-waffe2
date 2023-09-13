
(in-package :cl-user)

(defpackage :cl-waffe2/vm
  (:use
   :cl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl)
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
   #:wfop-args)
  (:export
   #:defpath
   #:FusionPathQuery
   #:make-query
   #:*user-defined-path-list*
   #:reset-all-path!))

(in-package :cl-waffe2/vm)

;; Provides cl-waffe2 VM, IR, ... without running (compile nil ..)


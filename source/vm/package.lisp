
(in-package :cl-user)

(defpackage :cl-waffe2/vm
  (:use
   :cl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl)
  (:export
   #:accept-instructions
   #:compile-forward-and-backward))

(in-package :cl-waffe2/vm)

;; Provides cl-waffe2 VM, IR, ... without running (compile nil ..)


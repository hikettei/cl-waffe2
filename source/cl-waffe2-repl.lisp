
(in-package :cl-user)

(defpackage :cl-waffe2-repl
  (:nicknames #:cl-waffe2-user)
  (:documentation "An playground place of cl-waffe2")
  (:use
   :cl
   :cl-user
   :common-lisp-user
   :cl-waffe2
   :cl-waffe2/vm
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl
   :cl-waffe2/distributions
   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.lisp
   :cl-waffe2/backends.jit.cpu
   :cl-waffe2/backends.aten
   :cl-waffe2/threads
   :cl-waffe2/nn
   :cl-waffe2/optimizers))




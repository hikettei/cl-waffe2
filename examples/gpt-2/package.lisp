
(in-package :cl-user)

(defpackage :gpt-2-example
  (:use
   :cl
   :alexandria
   :cl-waffe2
   :cl-waffe2/vm
   :cl-waffe2/nn
   :cl-waffe2/base-impl
   :cl-waffe2/distributions
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes

   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.lisp)
  (:export
   #:launch-repl))



(in-package :cl-user)

(defpackage :cl-waffe2
  (:use
   :cl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl
   :cl-waffe2/distributions
   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.lisp))


;; ./backends/cpu/
;; ./backends/cuda/
;; ./backends/metal/

;; ./nn/
;; ./optimizers
;; ./distrivutions

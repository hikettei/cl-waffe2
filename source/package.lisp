
(in-package :cl-user)

(defpackage :cl-waffe2
  (:use
   :cl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl
   :cl-waffe2/distributions
   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.lisp)
  (:export
   #:with-config
   #:with-dtype
   #:with-column-major
   #:with-row-major
   #:with-cpu
   ;;#:with-cuda
   #:with-build))

(in-package :cl-waffe2)

;; Export Most Basic APIs?

;;
;; :cl-waffe2/base-impl's all exported APIs
;; :cl-waffe2/distributions
;; :cl-waffe2/backends.lisp::CPUTensor, ScalarTensor etc..

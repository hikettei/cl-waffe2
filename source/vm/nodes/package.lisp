
(in-package :cl-user)

(defpackage :cl-waffe2/vm.nodes
  (:use :cl :cl-ppcre :alexandria)
  (:export
   #:defnode
   #:define-impl))
;; Export: defnode define-impl

(in-package :cl-waffe2/vm.nodes)

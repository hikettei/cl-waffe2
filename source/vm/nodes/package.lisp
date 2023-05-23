
(in-package :cl-user)

(defpackage :cl-waffe2/vm.nodes
  (:use :cl :cl-ppcre :alexandria)
  (:export
   #:*using-tensor*
   #:with-devices
   #:with-single-device
   #:*facet-monopoly-mode*)
  (:export
   #:defnode
   #:define-impl))
;; Export: defnode define-impl

(in-package :cl-waffe2/vm.nodes)

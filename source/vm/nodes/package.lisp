
(in-package :cl-user)

(defpackage :cl-waffe2/vm.nodes
  (:use :cl :cl-ppcre :alexandria)
  (:import-from :cl-waffe2/vm.generic-tensor
		#:*using-backend*
		#:shape
		#:tensor-prev-state
		#:tensor-prev-form
		#:tensor-variables
		#:make-tensor
		#:shaping-error)
  (:export
   #:with-devices
   #:with-single-device
   #:*facet-monopoly-mode*)
  (:export
   #:defnode
   #:define-impl))
;; Export: defnode define-impl

(in-package :cl-waffe2/vm.nodes)

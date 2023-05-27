
(in-package :cl-user)

(defpackage :cl-waffe2/vm.nodes
  (:use :cl :cl-ppcre :alexandria)
  (:import-from :cl-waffe2/vm.generic-tensor
		#:*using-backend*
		#:shape
		#:tensor-backward
		#:tensor-variables
		#:tensor-state
		#:tensor-out-n
		#:tensor-vec
		#:make-tensor
		#:shaping-error
		#:shape-equal
		#:make-statecontainer)
  (:export
   #:forward
   #:backward)
  
  (:export
   #:with-devices
   #:with-single-device
   #:*facet-monopoly-mode*)
  (:export
   #:defnode
   #:define-impl))
;; Export: defnode define-impl

(in-package :cl-waffe2/vm.nodes)

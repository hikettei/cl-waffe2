
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
		#:tensor-view
		#:make-tensor
		#:make-input
		#:shaping-error
		#:shape-equal
		#:order
		#:dtype
		#:make-statecontainer
		#:*no-grad*
		#:tensor-flexible-p
		#:with-no-grad)
  (:export
   #:forward
   #:backward
   #:node-passed-p
   #:ignore-shape-error
   #:out-scalar-p
   #:with-instant-kernel
   #:with-shape-checkpoint
   #:make-errorpoint)

  (:export
   #:defmodel
   #:Composite
   #:call)

  (:export
   #:*no-grad*
   #:with-no-grad)
  
  (:export
   #:with-devices
   #:with-single-device
   #:*facet-monopoly-mode*)
  (:export
   #:defnode
   #:define-impl)
  ;; Reject-p-utils
  (:export
   #:supported-dtypes-are))
;; Export: defnode define-impl

(in-package :cl-waffe2/vm.nodes)



(deftype shape-checkpoint-state-t ()
  `(and keyword (member :forward :backward :moving)))

(defstruct (ShapeErrorPoint
	    (:conc-name checkpoint-)
	    (:constructor make-errorpoint (state node-at)))
  (state state :type shape-checkpoint-state-t)
  (node-at node-at :type (or null AbstractNode)))

(defparameter *shape-error-when* (make-errorpoint :forward nil)
  "This parameter indicates when the ShapeError occurred.
*shape-error-when* is a type of ShapeErrorPoint")

(defmacro with-shape-checkpoint ((state node) &body body)
  "Updates checkpoint for shapeerror.
   set node=nil to extend previous state's node"
  `(let ((*shape-error-when* (make-errorpoint ,state
					      (or ,node
						  (checkpoint-node-at *shape-error-when*)))))
     ,@body))



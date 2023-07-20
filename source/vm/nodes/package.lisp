
(in-package :cl-user)

(defpackage :cl-waffe2/vm.nodes.facets-tmp (:use :cl))

(defpackage :cl-waffe2/vm.nodes
  (:use :cl :cl-ppcre :alexandria)
  (:import-from :cl-waffe2/vm.generic-tensor
		#:AbstractTensor
		#:*using-backend*
		#:shape
		#:tensor-backward
		#:tensor-id
		#:tensor-variables
		#:tensor-state
		#:tensor-name
		#:tensor-out-n
		#:tensor-vec
		#:tensor-view
		#:requires-grad
		#:make-tensor
		#:make-input
		#:actual-shape
		#:set-input
		#:tensor-stride
		#:compile-option-t
		#:movetensor-p
		#:shaping-error
		#:tensor-protect-me
		#:shape-equal
		#:order
		#:dtype
		#:make-statecontainer
		#:tensor-projected-p
		#:*no-grad*
		#:tensor-flexible-p
		#:with-no-grad
		#:make-compiled-kernel
		#:system-lazy-set-save-for-backward
		#:system-lazy-read-save-for-backward
		#:scalar-p)
  (:export
   #:on-finalizing-compiling
   #:node-output-shape
   #:create-subscript-p
   #:composite-where
   #:define-composite-function
   #:forward
   #:backward
   #:expand-backward
   #:node-passed-p
   #:ignore-shape-error
   #:out-scalar-p
   #:with-instant-kernel
   #:with-shape-checkpoint
   #:make-errorpoint
   #:node-local-variables
   #:declare-local-variables)

  (:export
   #:define-static-node
   #:set-save-for-backward
   #:read-save-for-backward
   #:with-reading-save4bw
   #:with-setting-save4bw)

  (:export
   #:defmodel
   #:Composite
   #:composite-traced-p
   #:composite-input-size
   #:composite-output-size
   
   #:AbstractNode
   #:call
   #:on-print-object
   #:find-params)
  
  (:export
   #:with-devices
   #:with-single-device
   #:*facet-monopoly-mode*)
  (:export
   #:defnode
   #:define-impl
   #:define-and-impl-node)
  ;; Reject-p-utils
  (:export
   #:supported-dtypes-are)

  (:export
   #:InstantKernelNode))
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



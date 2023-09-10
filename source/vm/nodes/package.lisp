
(in-package :cl-user)

(defpackage :cl-waffe2/vm.nodes.facets-tmp (:use :cl))

(defpackage :cl-waffe2/vm.nodes
  (:use :cl :cl-ppcre :alexandria :bordeaux-threads)
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
		#:shape-with-broadcastable
		#:make-input
		#:actual-shape
		#:set-input
		#:tensor-stride
		#:detach-p
		#:compile-option-t
		#:movetensor-p
		#:shaping-error
		#:tensor-protect-me
		#:*ranked-loop-result-cacher*
		#:shape-equal
		#:order
		#:dtype
		#:make-statecontainer
		#:tensor-projected-p
		#:tensor-iid
		#:*no-grad*
		#:tensor-flexible-p
		#:with-no-grad
		#:make-compiled-kernel
		#:system-lazy-set-save-for-backward
		#:system-lazy-read-save-for-backward
		#:scalar-p)
  (:export
   ;; External JIT
   #:make-backward
   #:on-finalizing-compiling
   #:on-finished-compiling

   #:node-sv4bw
   #:node-output-shape
   #:create-subscript-p
   #:composite-where

   #:forward
   #:backward
   
   #:node-passed-p
   #:ignore-shape-error

   #:out-scalar-p

   #:with-instant-kernel
   #:with-shape-checkpoint

   #:make-errorpoint
   #:node-local-variables
   #:declare-local-variables)

  (:export
   #:set-save-for-backward
   #:read-save-for-backward
   #:with-reading-save4bw
   #:with-setting-save4bw)

  (:export
   #:defmodel
   #:defmodel-as
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
   #:define-and-impl-node

   #:define-impl-op
   #:define-op)
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




(in-package :cl-waffe2/vm.nodes)

;; Shaping Error etc...
;; TODO: Make much more obvious the content of errors, compared to Numpy/PyTorch.

(define-condition subscripts-format-error (simple-error)
  ((because :initarg :because)
   (target-symbol :initarg :target)
   (subscripts :initarg :subscript)
   (msg :initarg :msg :initform ""))
  (:documentation "The Condition subscripts-format-error occurs when cl-waffe couldn't interpret the given subscript because of invaild format.")
  (:report
   (lambda (c s)
     (case (slot-value c 'because)
       (:too-many
	(format s "The Keyword ~a can be supplied once at most in the subscript.
Subscript: ~a"
		(slot-value c 'target-symbol)
		(slot-value c 'subscripts)))
       (:not-found
	(format s "The Keyword ~a must be supplied at least once but not found.~%
Subscript: ~a"
		(slot-value c 'target-symbol)
		(slot-value c 'subscripts)))
       (t
	(format s "cl-waffe couldn't parse the given subscript because of ~a.
SubScript: ~a
At       : ~a
Content  : ~a"
		(slot-value c 'because)
		(slot-value c 'subscripts)
		(slot-value c 'target-symbol)
		(slot-value c 'msg)))))))
  
(define-condition subscripts-content-error (simple-error)
  ((msg :initarg :msg))
  (:report
   (lambda (c s)
     (format s "~a" (slot-value c 'msg)))))


(define-condition node-not-found (simple-error)
  ((node :initarg :node))
  (:report
   (lambda (c s)
     (format s "
forward:
  (forward (~a) ...)
            └── Couldn't find any implementation

  The AbstractNode ~a is undoubtedly declared by defnode, but cl-waffe2 couldn't find any valid implementation (given by define-impl) from current devices.

    
    Current Devices: ~a

=>
   1. Explict devices you want to use with the macro with-devices
   2. Add an implementation to the node with whichever you like: define-impl or define-impl-op
   ~a
~a"
	     (slot-value c 'node)
	     (slot-value c 'node)
	     *using-backend*
	     (if (subtypep (slot-value c 'node) 'cl-waffe2/base-impl::MatmulNode)
		 "3. OpenBLAS may not be loaded correctly if you want to use the CPUTensor backend."
		 "")
	     (if *facet-monopoly-mode*
		 "~%With *facet-monopoly-mode* = t, cl-waffe only uses the first-priority device."
		 "")))))


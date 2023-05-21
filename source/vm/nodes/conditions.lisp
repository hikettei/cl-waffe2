
(in-package :cl-waffe2/vm.nodes)

;; Shaping Error etc...

(define-condition Parse-Subscript-Error ())

(define-condition subscripts-format-error (simple-error)
  ((error-why :initarg :because)
   (target-symbol :initarg :target)
   (subscripts :initarg :subscript)
   (msg :initarg :msg))
  (:documentation "The Condition subscripts-format-error occurs when cl-waffe couldn't interpret the given subscript because of invaild format.")
  (:report
   (lambda (c s)
     (case (slot-value c 'error-why)
       (:too-many
	(format s "The Keyword ~a can be used at once in the subscript.
Subscript: ~a"
		(slot-value c 'target-symbol)
		(slot-value c 'subscripts))
	)
       (:not-found
	(format s "The Keyword ~a must be appeared at least once but not found.~%
Subscript: ~a"
		(slot-value c 'target-symbol)
		(slot-value c 'subscripts))
	)
       (t
	(format s "cl-waffe couldn't parse the given subscript because of ~a.
SubScript: ~a
At       : ~a
Content  : ~a"
		(slot-value c 'error-why)
		(slot-value c 'subscripts)
		(slot-value c 'target-symbol)
		(slot-value c 'msg)))))))
  
(define-condition subscripts-content-error (simple-error)
  ((msg :initarg :msg))
  (:report
   (lambda (c s)
     (format s "~a" (slot-value c 'msg)))))

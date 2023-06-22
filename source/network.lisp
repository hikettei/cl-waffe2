
(in-package :cl-waffe2)

;; network.lisp provides macros that used to simplify network definition.

(defmodel (Encapsulated-Node (self node-func)
	   :slots ((node-func :initarg :node-func))
	   :initargs (:node-func node-func)
	   :on-call-> ((self x)
		       (funcall (slot-value self 'node-func) x))))
  
(defmacro asnode (function)
  "
## [macro] asnode

(call (asnode #'!tanh) x)
"
  `(Encapsulated-Node ,function))

(defmacro call-> (input &rest nodes)
  "
## [macro] call->

(call-> x
       (slot-value self 'linear1)
       (asnode #'!tanh)
       (slot-value self 'linear2)"

  (let ((nodes (reverse nodes)))
    (labels ((expand-call-form (rest-inputs)
	       (if rest-inputs
		   `(call ,(car rest-inputs)
			  ,(expand-call-form (cdr rest-inputs)))
		   input)))

      (expand-call-form nodes))))


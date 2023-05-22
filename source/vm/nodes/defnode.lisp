
(in-package :cl-waffe2/vm.nodes)

;; Node, Tensorをgenericにする

(defmacro defnode ((abstract-name
		   (&rest constructor-arguments)
		    &key
		      (where t)
		      (slots nil)
		      (documentation ""))
		   &body constructor-body
		     &aux (subscript-p (gensym)))
  "The macro defnode helps you to describe how nodes are working.

abstract-name... The Common Name for your node.

Common Name
 |-> Sub Node(:backend = :cpu)
  ...

=================================================================
Shaping Format:

Ignore with t.

"

  `(prog1
       (defclass ,abstract-name (AbstractNode)
	 ,slots
	 (:documentation ,documentation))
     
     (defun ,abstract-name (,@(cdr constructor-arguments))
       (let* ((,subscript-p (create-subscript-p ,where))
	      (,(car constructor-arguments)
		(make-instance ',abstract-name
			       :function-node ,subscript-p)))
	 (declare (ignorable ,(car constructor-arguments)))
	 ,@constructor-body))))

(defmacro define-impl (abstract-name backend-name))

(define-impl (AddNode :device CPUTensor)
  :forward ((x)
	    )
  :backward ((dy)
	     ))

(defnode (AddNode (myself) ;; constructor-argumentを:where句で使えるように
	  :where `([~] [~] -> [~] where a = 0)
	  :documentation "The Node Addnode Provides ...")
  ;; When Initialized...
  ;; Here Follows Your Task
  )


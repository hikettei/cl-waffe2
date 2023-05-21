
(in-package :cl-waffe2/vm.nodes)

(defmacro defnode (abstract-name
		   (&rest initialize-arguments)
		   (&key (where t))
		   (&key (slots nil)
		         (documentation ""))
		   &body initialize-instance-body)
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
	 (:documentation ,documentation)
	 )

     (defmethod test-shape ()

       )
     
     (defmethod initialize-instance :before (,abstract-name ,@initialize-arguments)
       ,@initialize-instance-body)))


(defnode AddNode (x y) (:where t) ()
    addnode
    )


(defnode ReshapeNode (tensor &rest subscripts)
    (:where ([first-shape] -> [out-shape]
			   let
			     x = subscripts
			     first-shape = (shape tensor)))
    (:slots ((shape-state :type list))
     :documentation "Reshapes tensor")

  (setf (slot-value reshapenode 'shape-state) subscripts))

    
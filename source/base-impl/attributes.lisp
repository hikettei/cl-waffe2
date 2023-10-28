
(in-package :cl-waffe2/base-impl)

;; Abstract nodes can inherit these classes to determine which attributes to use for optimization

;; [TODO] (defclass AbstractMoveNode)
;; MoveScalarTensorNode ... should be pruned likewise MoveTensorNode

(defclass Loadp-Node () nil
  (:documentation "
Loads a system pointer from B to A:
A* = B* where A[~] B[~] -> A[~]
Accordingly, its forward definition is given as:
#'(lambda (from to)
    (setf (tensor-vec from) (tensor-vec to))
    from)"))

(defclass Loadp-Node-Rev () nil
  (:documentation "
#'(lambda (from to)
     (setf (tensor-vec to) (tensor-vec from))
     to)"))

(defmethod ensure-loadp ((node Loadp-Node) args out-to)
  "Ensures the node is declared as A* = B* forms.
FROM <- (FROM, TO)"
  (and
   (= (length out-to) 1)
   (= (length args)   2)))

(defclass Rebundant-Node () nil
  (:documentation "
AbstractNodes which extends this class is destinated to be eliminated when compiled and after backprop iseq is constructed.
"))

(defclass Load-MySelf-Node () nil
  (:documentation "
AbstractNodes which extends this class is interpreted as: lambda(x) (tensor-vec x)
"))



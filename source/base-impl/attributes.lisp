
(in-package :cl-waffe2/base-impl)

;; Abstract nodes can inherit these classes to determine which attributes to use for optimization

;; [TODO] (defclass AbstractMoveNode)
;; MoveScalarTensorNode ... should be pruned likewise MoveTensorNode

(defclass Loadp-Node () nil
  (:documentation "
Loads a system pointer from B to A:
A* = B* where A[~] B[~] -> A[~]"))

(defclass Rebundant-Node () nil
  (:documentation "
AbstractNodes which extends this class is destinated to be eliminated when compiled and after backprop iseq is constructed.
"))



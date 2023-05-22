
(in-package :cl-waffe2/vm.nodes)

(defclass AbstractNode ()
  ((function-node
    :initarg
    :function-node
    :reader abstractnode-node
    :type function) ;; [x y] [y z] -> [z x]
   (variables :initform nil :reader node-variables :writer set-variables :type list))
  (:documentation "The class AbstractNode is a fundamental object of describing computation nodes in cl-waffe.

AbstractNode must possess following:
   1. 遷移図
   2. Slots (for passing forward/backward)
   3. Variables (for building computation nodes)
   4. with-facet (facet ... ノードの様相)

backendのforward/backwardはAbstractNodeを継承して、定義する
"))

(defmethod test-and-forward-shape ((node AbstractNode) &rest previous-shape)
  ""
  (funcall (abstractnode-node node) previous-shape))

(defmethod forward :around ((node AbstractNode) &rest inputs)
  ;; Update Computation Nodes

  ;; Here do we (dolist (tensor inputs) (setf tensor.variables ...))
  (let ((out (multiple-value-list (call-next-method))))
    out))

(defmethod forward ((node AbstractNode) &rest inputs)
  (declare (ignore node inputs))
  ;; Describe More Errors.
  (error "Forward isn't implemented yet"))

(defmethod backward :around ((node AbstractNode) dy)
  (let ((out (multiple-value-list (call-next-method))))
    ;; update or lazy-evaluate
    out))

(defmethod backward ((node AbstractNode) dy)
  (error "Backward isn't implemented yet."))

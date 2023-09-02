
(in-package :cl-waffe2/vm.generic-tensor)

;; Utils for Graph

(defun deterministic-p (tensor)
  "Returns t if tensor's node is deterministic
[Any-Previous-Node]
    |
[AnyNode] <- The Given Tensor
    |"
  (declare (type AbstractTensor tensor))
  (= (length (tensor-variables tensor)) 1))

(defun non-deterministic-p (tensor)
  "Returns t if tensor's node is non-deterministic
[Node1] [Node2] ...
    |------|
[AnyNode] <- The Given Tensor
    |"
  (declare (type AbstractTensor tensor))
  (> (length (tensor-variables tensor)) 1))

(deftype node-state-t ()
  "The type node-state-t indicates the keywords that used to express node's transmission state."
  `(member :deterministic :non-deterministic))

(declaim (ftype (function (AbstractTensor) node-state-t) node-state))
(defun node-state (tensor)
  (if (deterministic-p tensor)
      :deterministic
      :non-deterministic))

(defun movetensor-p (node)
  (or (subtypep (class-of node) (find-class 'cl-waffe2/base-impl:MoveTensorNode))
      (subtypep (class-of node) (find-class 'cl-waffe2/base-impl::MoveScalarTensorNode))))

(defmacro ignore-me? (node)
  `(cl-waffe2/base-impl:movetensor-ignore-me ,node))

(defun tensor-attribute (tensor)
  "Return: (member :chain :input)"
  (declare (type AbstractTensor tensor))
  (let ((name (tensor-name tensor)))
    (typecase name
      (string
       (if (eql (tensor-facet tensor) :input)
	   :chain
	   :input)) ;; :chain = auto-generated
      (T :input))))

(defun trace-and-explore-nodes! (out-tensor)
  (error "to be deleted"))

(defun trace-and-optimize-node! (out-tensor major n-cores)
  (error "to be deleted"))

(defun optimize-computation-node! (out-tensor major n-cores)
  (error "to be deleted")) 

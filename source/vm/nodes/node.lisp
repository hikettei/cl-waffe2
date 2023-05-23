
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

;; TODO: Under here.
(defmethod test-and-forward-shape ((node AbstractNode) &rest previous-shape)
  ""
  (funcall (abstractnode-node node) previous-shape))

(defgeneric forward  (node &rest inputs))
(defgeneric backward (node dy))

(defmethod forward :around ((node AbstractNode) &rest inputs)
  ;; Update Computation Nodes

  ;; Here do we (dolist (tensor inputs) (setf tensor.variables ...))
  (let* ((transition-function (abstractnode-node node)))
    (multiple-value-bind (result-shape detected-errors) (funcall transition-function (loop for i in inputs collect (shape i)))
      
      (when detected-errors
	(error "Errors detected: ~a" detected-errors))

      (let* ((form-expanded (multiple-value-list (call-next-method)))
	     (next-tensor
	       (loop for shape    in result-shape
		     for out-form in form-expanded
		     collect (let ((next-tensor (make-tensor shape)))
			       (setf (tensor-prev-form  next-tensor) out-form)
			       (setf (tensor-prev-state next-tensor) node)
			       (setf (tensor-variables  next-tensor) inputs)
			       next-tensor))))
	(apply #'values next-tensor)))))

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


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

(defun describe-problems (error-node detected-errors)
  (shaping-error
   "Couldn't step forward because of shape-error.
At: ~a
Here's a list of reports.

1. ~a

~a
~a"
   error-node
   (car detected-errors)
   (if (cdr detected-errors)
       "Also, these reports could be helpful for you (calculated ignoring the first errors.)"
       "")
   (with-output-to-string (out)
     (loop for err in (cdr detected-errors)
	   for n upfrom 1
	   do (format out "~%~%~a. ~a" n err)))))

(defgeneric forward  (node &rest inputs))
(defgeneric backward (node dy))

(defmethod forward :around ((node AbstractNode) &rest inputs)
  ;; Update Computation Nodes

  (let* ((transition-function (abstractnode-node node)))
    (multiple-value-bind (result-shape detected-errors) (funcall transition-function (loop for i in inputs collect (shape i)))
      
      (when detected-errors
	(describe-problems node detected-errors))

      ;; num of iters depends on how many out declared in node.
      (let* ((out-form (call-next-method))
	     (next-tensor
	       (loop for shape    in result-shape
		     for nth-arg  upfrom 0
		     collect (let ((next-tensor (make-tensor shape)))
			       ;; (values x y) <- 二回重複して実行されない？
			       ;; out-formを補完するためのデータ型を作る
			       (setf (tensor-prev-form  next-tensor) `(nth ,nth-arg (multiple-value-list (progn ,out-form))))
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

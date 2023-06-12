
(in-package :cl-waffe2/base-impl)

(defnode (MoveTensorNode (myself)
	  :where (A[~] B[~] -> A[~])
	  :slots ((ignore-me :initform nil :accessor movetensor-ignore-me :type boolean)
		  (save-for-backward :initform nil :accessor movetensor-save-for-backward :type boolean)) ;; when t, ignored.
	  :backward ((self dout dx dy)
		     (declare (ignore dx))
		     (let ((dy-out
			     (if (and
				  (eql (tensor-attribute dy) :chain)
				  (movetensor-ignore-me self))
				 dout
				 (!copy dout))))
		       (values dout dy-out)))
	  :documentation "
The Node MoveTensorNode must satisfy the following behaviours:

Forward:
1. If ignore-me is t, return the given value itself.
2. Otherwise, move x <- y.

Note that until (tensor-vec) is called, x is never allocated.

The option ignore-me can be accessed by the function (movetensor-ignore-me MoveTensorNode)"))

(defnode (MoveScalarTensorNode (myself)
	  :out-scalar-p t
	  :where (A[scal] B[scal] -> A[scal] where scal = 1)
	  :backward ((self dout dx dy)
		     (declare (ignore dx))
		     (let ((dy-out
			     (if (and
				  (eql (tensor-attribute dy) :chain)
				  (movetensor-ignore-me self))
				 dout
				 (!copy dout))))
		       (values dout dy-out)))))

(define-impl (MoveScalarTensorNode :device ScalarTensor)
	     :forward ((self x y)
		       `(setf (tensor-vec ,x) (tensor-vec ,y))))

;; TODO: Move For Scalar
(defun !move (place tensor)
  "TODO: DOCSTRING"
  (if (and (scalar-p place)
	   (scalar-p place))
      (forward (MoveScalarTensorNode) place tensor)
      (forward (MoveTensorNode) place tensor)))

(defun !copy (tensor)
  "TODO: DOCSTRING"
  (let ((out (make-input (shape tensor) nil
			 :scalar-p (scalar-p tensor)
			 :dtype (dtype tensor)
			 :order (order tensor))))
    (!move out tensor)))

(defnode (ViewTensorNode (myself subscripts result1 before1)
	  :slots ((subscripts :initarg :subscripts))
	  :where (A[result] B[before] -> A[result] where result = result1 before = before1))
  (setf (ignore-shape-error myself) t))

(define-impl (ViewTensorNode)
	     :forward
	     ((self viewed-tensor old)
	      (declare (ignore old))
	      `(progn
		 ,viewed-tensor))
	     :backward
	     ((self dout dx dy)
	      (let ((out-sub (tensor-view dy))
		    (inp-sub (slot-value self 'subscripts)))
		(values
		 nil
		 (!move
		  dy
		  (apply
		   #'!view
		   (!move dx (apply #'!view dout inp-sub))
		   out-sub))))))

(defun !view (tensor &rest subscripts)
  "TODO: DOC"
  (let ((out (apply #'cl-waffe2/vm.generic-tensor::view tensor subscripts)))
    ;; Update Chains
    (forward (ViewTensorNode subscripts (shape out) (shape tensor)) out tensor)))

(defnode (ReshapeTensorNode (self before shape)
	  :where (A[shape-before] B[shape-after] -> B[shape-after] where shape-before = before shape-after = shape)
	  :slots ((before :initarg :before :reader reshapenode-shape))
	  :backward ((self dout dx dy)
		     (declare (ignore dx dy))
		     (values (apply #'!reshape dout (reshapenode-shape self)) nil))
	  :documentation "")
  (setf (ignore-shape-error self) t))

(define-impl (ReshapeTensorNode :device t)
	     :save-for-backward (t) ;; =T is necessary not to delete MoveTensorNode.
	     :forward ((self x y)
		       `(progn
			  (setf (tensor-vec ,y) (tensor-vec ,x))
			  ,y)))

(defun !reshape (tensor &rest shapes)
  "Reshapes the tensor.
TODO:DOCSTRING"
  (declare (type AbstractTensor tensor))
  (assert (= (apply #'* (shape tensor))
	     (apply #'* shapes))
	  nil
	  "Total size doesn't match")
  (let ((result (make-input shapes nil
			    :dtype (dtype tensor)
			    :order (order tensor))))
    ;; (!view tensor `(2 4) `(2 4)) -> Copy
    ;; (!view tensor  0 t t t)
    (if (tensor-projected-p tensor)
	(forward (ReshapeTensorNode (shape tensor) shapes) (!copy tensor) result)
	(forward (ReshapeTensorNode (shape tensor) shapes) tensor result))))

;; The definition of value node is dynamically changed and redefined.
;; Forward  -> All The Previous Forward Steps
;; Backward -> All The Previous Backward Steps.

;; We can also add: Proceed-Auto

(defnode (ProceedNode (myself &key (measure-time nil))
	  :where (A[~] -> B[~])
	  :slots ((measure-time :initarg :measure-time :reader measure-time-p)
		  (backward :accessor proceed-backward)
		  (result   :accessor proceed-result))
	  :documentation "ProceedNode is a special node which takes all the previous computation node before tensor."))

(define-impl (ProceedNode :device t)
	     :save-for-backward (nil)
	     :forward ((self x)
		       (multiple-value-bind (fw bw vars params) (build x)
			 (declare (ignore vars params))
			 ;; Vars/Params will be tracked by other build.
			 (setf (proceed-backward self) bw)
			 (if (measure-time-p self)
			     (setf (proceed-result self) (time (funcall fw)))
			     (setf (proceed-result self) (funcall fw)))
			 (setf (out-scalar-p self) (scalartensor-p x))
			 `(progn ,x)))
	     :backward ((self dout dx)
			(declare (ignore dx))
			(let ((bw (proceed-backward self)))
			  (values
			   (with-instant-kernel dout
			     `(and
			       ,(if (measure-time-p self)
				    `(time (funcall ,bw))
				    `(funcall ,bw))
			       ;; Delete Gradients.
			       (!mul 0 ,dout)))))))

;; TODO: ProceedNode for several outputs
(defun proceed (tensor &key (measure-time nil))
  "The function proceed invokes special node, ProceedNode, which takes all the previous computation node before tensor, returning the result of it.
The backward is created with the previous node.

This function will be useful especially when debugging on REPL.

Also, using (with-dynamically-mode) will invoke this function every time forward invoked."
  (let* ((node (ProceedNode :measure-time measure-time))
	 ;; Previous Node is already compiled, so detach tensor from nodes.
	 (out (forward node tensor)))
    
    ;; Cut off previous backwards
    (setf (tensor-backward tensor) nil)

    ;; Out is still unallocated, so set the result.
    (embody-actual-tensor out (proceed-result node))
    out))

(defun proceed-time (tensor)
  "Proceed with time macro"
  (proceed tensor :measure-time t))

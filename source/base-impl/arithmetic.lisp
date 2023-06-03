
(in-package :cl-waffe2/base-impl)

;;
;; Add: define-doc
;; Matrix-Matrix Operation:
;;  (!matrix-add ) (!matrix-sub ) (!matrix-mul ) (!matrix-div )
;; Correspondng-Nodes are AddNode SubNode MulNode DivNode
;;
;; Scalar-Matrix Operation:
;;  (!scalar-add ) (!scalar-sub ) ...
;; Corresponding-Nodes are ScalarAdd ScalarSub...
;;
;; general-purpose function: !add !sub !mul !div.

;; ===============================================================
;; Defnode Parts
(macrolet ((define-arithmetic-node (name document1 document2)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defnode (,name (myself)
			  :where `([~] [~] -> [~])
			  :documentation ,(format nil "~a is a node which computes following operation element-wise.
Let X and Y be a given arguments and both are matrix.
   X <- X ~a Y" document1 document2))))))
  (define-arithmetic-node AddNode "AddNode" "+")
  (define-arithmetic-node SubNode "SubNode" "-")
  (define-arithmetic-node MulNode "MulNode" "*")
  (define-arithmetic-node DivNode "DivNode" "/"))


(macrolet ((define-scalar-mat-node (name document1 document2)
	     `(progn
		(export ',name)
		(defnode (,name (myself)
			  :where `([scal] [~] -> [~] where scal = 1)
			  :documentation ,(format nil
						  "~a is a node which computes following operation element-wise.
Let X be a given matrix and S be a given scalar.
    X <- scalar ~a X" document1 document2))))))
  (define-scalar-mat-node
      ScalarAddNode
    "ScalarAddNode"
    "+")

  (define-scalar-mat-node
      ScalarMulNode
    "ScalarMulNode"
    "*"))
;; ===============================================================


;; 「!」 key can be hit in both the JP and EN sequences without breaking the home position.
;; Keep in mind: https://arxiv.org/pdf/1201.6035.pdf

;; ===============================================================
;; Defun Parts
(macrolet ((define-arithmetic-node-caller (name node-name ops prep)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y)
		  ,(format nil "The function ~a ~a X ~a Y element-wise.

Side Effects: None.

Note that the operation is automatically replaced into in-place operation."
			   (symbol-name name) ops prep)
		  ;; Note: !copy is only needed when backward will be used.
		  ;; FIXME: The usage of forward below seems a little tricky
		  (forward (,node-name)
			   (!copy x)
			   (if *no-grad*
			       y
			       (!copy y)))))))
  (define-arithmetic-node-caller
      !matrix-add
    AddNode
    "adds"
    "and")
  (define-arithmetic-node-caller
      !matrix-sub
    SubNode
    "substracts"
    "by")
  (define-arithmetic-node-caller
      !matrix-mul
    MulNode
    "multiplies"
    "and")
  (define-arithmetic-node-caller
      !matrix-div
    DivNode
    "divides"
    "by"))

(macrolet ((define-scalar-mat-node-caller (name node-name document)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (scalar x)
		  ,document
		  (forward (,node-name) scalar (!copy x))))))
  (define-scalar-mat-node-caller
      !scalar-add ScalarAddNode
    "DOC")
  (define-scalar-mat-node-caller
      !scalar-mul ScalarMulNode
    "DOC"))

(with-export !scalar-sub
  (defun !scalar-sub (scalar x)
    "DOC"
    (!scalar-add (- scalar) x)))

(with-export !scalar-div
  (defun !scalar-div (scalar x)
    "DOC"
    (!scalar-mul (/ scalar) x)))

(defun scalar-p (tensor)
  (subtypep (class-of tensor) 'cl-waffe2/vm.generic-tensor:ScalarTensor))

(macrolet ((define-arith-function (name
				   invertor
				   scalar-and-scalar-operation
				   scalar-operation
				   matrix-operation)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y)
		  ""
		  (cond
		    ((and (scalartensor-p x)
			  (scalartensor-p y))
		     (,scalar-and-scalar-operation x y))
		    ((scalartensor-p x)
		     ;; X is scalar, Y is matrix.
		     (,scalar-operation x y))
		    ((scalartensor-p y)
		     (,scalar-operation y (,invertor x)))
		    (T
		     (,matrix-operation x y)))))))

  )

;; ===============================================================
;; Scalar-And-Scalar Defnode And Functions.
;; ===============================================================

(macrolet ((define-sas-node (name)
	     `(defnode (,name (myself)
			:where `([scal] [scal] -> [scal] where scal = 1)))))
  (define-sas-node ScalarAndScalarAdd)
  (define-sas-node ScalarAndScalarSub)
  (define-sas-node ScalarAndScalarMul)
  (define-sas-node ScalarAndScalarDiv))
		  

(define-impl (ScalarAndScalarAdd :device ScalarTensor)
	     :forward ((self x y)
		       `(and
			 (incf (tensor-vec ,x) (tensor-vec ,y))
			 ,x))
	     :backward ((self dout dx dy)
			(values dout dout)))


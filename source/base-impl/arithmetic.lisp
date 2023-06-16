
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

;; Utils
(defun number->stensor (scalar tensor)
  "This function always returns ScalarTensor, whenever scalar is number or ScalarTensor. tensor[number or AbstractTensor] is used in order to determine the dtype of scalar."
  (if (numberp scalar)
      (make-tensor scalar :dtype (if (numberp tensor)
				     (dtype-of tensor)
				     (dtype tensor)))
      scalar))

;; ===============================================================
;; Defnode Parts
;; ===============================================================
(macrolet ((define-arithmetic-node (name document1 document2 &optional backward)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defnode (,name (myself dtype)
			  :where (A[~] B[~] -> A[~])
			  :backward ,backward
			  :documentation ,(format nil "~a is a node which computes following operation element-wise.
Let X and Y be a given arguments and both are matrix.
   X <- X ~a Y" document1 document2))))))
  (define-arithmetic-node AddNode "AddNode" "+"
    ((self dout dx dy)
     (values (!move dx dout) (!move dy dout))))
  (define-arithmetic-node SubNode "SubNode" "-"
    ((self dout dx dy)
     (values (!move dx dout) (!move dy (!mul -1 dout)))))
  (define-arithmetic-node MulNode "MulNode" "*"
    ((self dout dx dy)
     (values (!mul dout dy) (!mul dout dx))))
  (define-arithmetic-node DivNode "DivNode" "/"
    ((self dout dx dy)
     ;; ∂/∂x = 1/x
     ;; ∂/∂y = -x/y^2
     (values
      (!div dout dy)
      (!div (!mul dx (!mul -1 dout))
	    (!square dy))))))

(macrolet ((define-scalar-mat-node (name document1 document2 &optional backward)
	     `(progn
		(export ',name)
		(defnode (,name (myself dtype)
			  :where (A[scal] B[~] -> B[~] where scal = 1)
			  :backward ,backward
			  :documentation ,(format nil
						  "~a is a node which computes following operation element-wise.
Let X be a given matrix and S be a given scalar.
    X <- scalar ~a X" document1 document2))))))
  (define-scalar-mat-node
      ScalarAdd
    "ScalarAdd"
    "+"
    ((self dout dx dy)
     ;; dx <- scalar
     ;; dy <- matrix
     ;; A+=scal.view(A.shape),
     (declare (ignore dx dy))
     (values
      (->scal (!mean dout))
      dout)))
  
  (define-scalar-mat-node
      ScalarSub
    "ScalarSub"
    "-"
    ((self dout dx dy)
     (declare (ignore dx dy))
     (values
      (->scal (!mean dout))
      (!mul -1.0 dout))))

  (define-scalar-mat-node
      ScalarMul
    "ScalarMul"
    "*"
    ((self dout dx dy)
     ;; dx ... scalar
     ;; dy ... matrix

     (values
      (->scal (!mean (!mul dy dout)))
      (!mul dout dx))))

  (define-scalar-mat-node
      ScalarDiv
    "ScalarDiv"
    "/"
    ((self dout dx dy)
     ;; dx ... scalar
     ;; dy ... matrix
     (values
      (->scal (!mean (!div dout dy)))
      (!div (!mul dx (!mul -1 dout))
	    (!square dy))))))

;; ===============================================================
;; Defun Parts
;; ===============================================================
(macrolet ((define-arithmetic-node-caller (name node-name ops prep)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y)
		  ,(format nil "The function ~a ~a X ~a Y element-wise.

Side Effects: None.

Note that the operation is automatically replaced into in-place operation."
			   (symbol-name name) ops prep)
		  ;; Note: !copy is only needed when backward will be used.
		  ;; Note: The usage of forward below seems a little tricky
		  (forward (,node-name (dtype x))
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

;; update docs
(macrolet ((define-scalar-mat-node-caller (name node-name document)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (scalar x)
		  ,document
		  (forward (,node-name (dtype x))
			   (number->stensor scalar x) (!copy x))))))
  (define-scalar-mat-node-caller
      !scalar-add ScalarAdd
    "X += scalar")
  (define-scalar-mat-node-caller
      !scalar-sub ScalarSub
    "X -= scalar")
  (define-scalar-mat-node-caller
      !scalar-mul ScalarMul
    "X *= scalar")
  (define-scalar-mat-node-caller
      !scalar-div ScalarDiv
    "X /= scalar"))

;; ===============================================================
;; Scalar-And-Scalar Defnode And Functions.
;; ===============================================================

(macrolet ((define-sas-node (name)
	     `(defnode (,name (myself)
			:out-scalar-p t
			:where (A[scal] B[scal] -> A[scal] where scal = 1)))))
  (define-sas-node ScalarAndScalarAdd)
  (define-sas-node ScalarAndScalarSub)
  (define-sas-node ScalarAndScalarMul)
  (define-sas-node ScalarAndScalarDiv))

(define-impl (ScalarAndScalarAdd :device ScalarTensor)
	     :forward ((self x y)
		       (let ((t1 (dtype->lisp-type (dtype x)))
			     (t2 (dtype->lisp-type (dtype y))))
			 `(and
			   (setf (the ,t1 (tensor-vec ,x))
				 (+ (the ,t1 (tensor-vec ,x))
				    (the ,t2 (tensor-vec ,y))))
			   ,x)))
	     :backward ((self dout dx dy)
			(declare (ignore dx dy))
			(values dout dout)))

(define-impl (ScalarAndScalarSub :device ScalarTensor)
	     :forward ((self x y)
		       (let ((t1 (dtype->lisp-type (dtype x)))
			     (t2 (dtype->lisp-type (dtype y))))
			 `(and
			   (setf (the ,t1 (tensor-vec ,x))
				 (- (the ,t1 (tensor-vec ,x))
				    (the ,t2 (tensor-vec ,y))))
			   ,x)))
	     :backward ((self dout dx dy)
			(declare (ignore dx dy))
			(values dout (!sas-mul -1 dout))))

(define-impl (ScalarAndScalarMul :device ScalarTensor)
	     :save-for-backward (t t)
	     :forward ((self x y)
		       (let ((t1 (dtype->lisp-type (dtype x)))
			     (t2 (dtype->lisp-type (dtype y))))
			 `(and
			   (setf (the ,t1 (tensor-vec ,x))
				 (the ,t1
				      (* (the ,t1 (tensor-vec ,x))
					 (the ,t2 (tensor-vec ,y)))))
			   ,x)))
	     :backward ((self dout dx dy)
			(values (!sas-mul dy dout)
				(!sas-mul dx dout))))

(define-impl (ScalarAndScalarDiv :device ScalarTensor)
	     :save-for-backward (t t)
	     :forward ((self x y)
		       (let ((t1 (dtype->lisp-type (dtype x)))
			     (t2 (dtype->lisp-type (dtype y))))
			 `(and
			   (setf (the ,t1 (tensor-vec ,x))
				 (/ (the ,t1 (tensor-vec ,x))
				    (the ,t2 (tensor-vec ,y))))
			   ,x)))
	     :backward ((self dout dx dy)
			;;
			;; ∂/∂x = 1/y
			;; ∂/∂y = -x/y^2
			(values (!sas-div dout dy)
				(!sas-div
				 (!sas-mul dx (!sas-mul -1 dout))
				 (!square dy)))))

(macrolet ((define-sas-op (name node-name)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y)
		  "TODO: Docstring"
		  (forward (,node-name)
			   (!copy (number->stensor x y))  ;; Returns X
			   (!copy (number->stensor y x))) ;; Returns Y
		  ))))
  (define-sas-op !sas-add ScalarAndScalarAdd)
  (define-sas-op !sas-sub ScalarAndScalarSub)
  (define-sas-op !sas-mul ScalarAndScalarMul)
  (define-sas-op !sas-div ScalarAndScalarDiv))

;; ===============================================================
;; Defines general-purpose functions.
;; ===============================================================


(defun scalartensor-p (tensor)
  (scalar-p tensor))

(macrolet ((define-arith-function (name
				   invertor
				   scalar-and-scalar-operation
				   scalar-operation
				   matrix-operation)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y)
		  "TODO: Document, Broadcast-auto"
		  (let ((x (number->stensor x y))  ;; Returns X
			(y (number->stensor y x))) ;; Returns Y
		    (cond
		      ((and (scalar-p x)
			    (scalar-p y))
		       (,scalar-and-scalar-operation x y))
		      ((scalar-p x)
		       ;; X is scalar, Y is matrix.
		       (,scalar-operation x y))
		      ((scalar-p y)
		       (,scalar-operation y (,@invertor x)))
		      (T
		       (,matrix-operation x y))))))))
  (define-arith-function
      !add (progn) !sas-add !scalar-add !matrix-add)
  (define-arith-function
      !sub (!mul -1) !sas-sub !scalar-sub !matrix-sub)
  (define-arith-function
      !mul (progn) !sas-mul !scalar-mul !matrix-mul)
  (define-arith-function
      !div (!div 1) !sas-div !scalar-div !matrix-div))


;; ===============================================================
;; Destructive Functions Family: A+=B A-=B A*=B A/=B
;; ===============================================================

(macrolet ((define-darith-function (name
				    matrix-operation)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (A B)
		  "TODO: Docstring"
		  (declare (type AbstractTensor A B))
		  (assert (or (not (scalar-p A))
			      (not (scalar-p B)))
			  nil
			  "Assertion Failed with A and B both aren't scalar.")
		  (forward (,matrix-operation (dtype A)) A B)))))
  (define-darith-function A+=B AddNode)
  (define-darith-function A-=B SubNode)
  (define-darith-function A*=B MulNode)
  (define-darith-function A/=B DivNode))

;; ===============================================================
;; Destructive Functions Family: A+=scal A-=scal A*=scal A/=scal
;; ===============================================================

(macrolet ((define-darith-function (name
				    matrix-operation)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (A scalar)
		  "TODO: Docstring"
		  (if (numberp scalar)
		      (forward (,matrix-operation (dtype A)) (make-tensor scalar) A)
		      (forward (,matrix-operation (dtype A)) scalar A)))))
	   (define-darith-function1 (name broadcast op op1 arg)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (A scalar)
		  "TODO: Docstring"
		  (if (numberp scalar)
		      (,broadcast (,op  (number->stensor scalar A)) A)
		      (,broadcast (,op1 ,arg (number->stensor scalar A)) A))))))
  (define-darith-function  A+=scal ScalarAdd)
  (define-darith-function1 A-=scal A+=scal - !mul -1)
  (define-darith-function  A*=scal ScalarMul)
  (define-darith-function1 A/=scal A*=scal / !div 1))


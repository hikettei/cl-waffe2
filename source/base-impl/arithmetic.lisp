
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
				     (if (typep tensor 'AbstractTensor)
					 (dtype tensor)
					 (dtype-of scalar))))
      (if (symbolp scalar)
	  (make-tensor scalar :dtype :uint32)
	  scalar)))

;; ===============================================================
;; Defnode Parts
;; ===============================================================
(macrolet ((define-arithmetic-node (name document1 document2 sv4bw &optional backward)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defnode (,name (myself dtype)
			  ;; In backward:
			  ;;       dx   dy     dout
			  :where (A[~] B[~] -> A[~])
			  :save-for-backward ',sv4bw
			  :backward ,backward
			  :documentation ,(format nil "`~a` is a node which computes following operation element-wise.

Let X and Y be a given arguments and both are matrix.

```math
X\\gets{X ~a Y}
```

### Constructor

```
(~a dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)

" document1 document2 document1))))))
  (define-arithmetic-node AddNode "AddNode" "+" nil
    ((self dout dx dy)
     (declare (ignore dx dy))
     (values dout dout)))
  (define-arithmetic-node SubNode "SubNode" "-" nil
    ((self dout dx dy)
     (declare (ignore dx dy))
     (values dout (!mul -1 dout))))
  (define-arithmetic-node MulNode "MulNode" "*" (t t)
    ((self dout dx dy)
     (values
      (!mul dout dy)
      (!mul dout dx))))
  (define-arithmetic-node DivNode "DivNode" "/" (t t)
    ((self dout dx dy)
     ;; ∂/∂x = 1/x
     ;; ∂/∂y = -x/y^2
     (values
      (!div dout dy)
      (!div (!mul dx (!mul -1 dout))
	    (!square dy))))))

(defnode (InverseTensorNode (myself dtype)
	  :where (A[~] -> A[~])
	  :save-for-backward (t)
	  :backward ((self dout dx)
		     (values (!div (!mul -1 dout) (!square dx))))
	  :documentation "InverseTensorNode is a node which computes following operation element-wise

```math
A\\gets{1 / A}
```

### Constructor

```
(InverseTensorNode dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)

"))

(macrolet ((define-scalar-mat-node (name document1 document2 sv4bw &optional backward)
	     `(progn
		(export ',name)
		(defnode (,name (myself dtype)
			  :where (A[~] Scalar[scal] -> A[~] where scal = 1)
			  :save-for-backward ',sv4bw
			  :backward ,backward
			  :documentation ,(format nil
						  "~a is a node which computes following operation element-wise.

Let X be a given matrix and S be a given scalar.

```math
X\\gets{X ~a scalar}
```

### Constructor

```
(~a dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)
" document1 document2 document1))))))
  (define-scalar-mat-node
      ScalarAdd
    "ScalarAdd"
    "+"
    nil
    ((self dout dx dy)
     ;; dx <- matrix
     ;; dy <- scalar
     ;; A+=scal.view(A.shape),
     (declare (ignore dx dy))
     (values
      dout
      (->scal (!mean dout)))))
  
  (define-scalar-mat-node
      ScalarSub
    "ScalarSub"
    "-"
    nil
    ((self dout dx dy)
     (declare (ignore dx dy))
     (values
      dout
      (->scal (!mul -1.0 (!mean dout))))))

  (define-scalar-mat-node
      ScalarMul
    "ScalarMul"
    "*"
    (t t)
    ((self dout dx dy)
     ;; dx ... matrix
     ;; dy ... scalar

     (values
      (!mul dout dy)
      (->scal (!mean (!mul dx dout))))))
  
  (define-scalar-mat-node
      ScalarDiv
    "ScalarDiv"
    "/"
    (t t)
    ((self dout dx dy)
     ;; dx ... scalar
     ;; dy ... matrix
     ;; out = 1/dx * dy
     (values
      (!div dout dy)
      (->scal (!mean (!div (!mul dx (!mul -1 dout)) (!square dy))))))))

;; ===============================================================
;; Defun Parts
;; ===============================================================
(macrolet ((define-arithmetic-node-caller (name node-name ops prep f)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y)
		  ,(format nil "
## [function] ~(~a~)

```lisp
(~(~a~) x y)
```

The function `~(~a~)` calls `~a` and ~a X ~a Y element-wise, returning a new tensor.

```math
X_{copy}\\gets{X ~a Y}
```

### Inputs

`X` and `Y` must be a AbstractTensor (not a ScalarTensor), with the same shape.

### SideEffects

None.
"
			   (symbol-name name)
			   (symbol-name name)
			   (symbol-name name)
			   (symbol-name node-name)
			   ops
			   prep
			   f)
		  ;; Note: !copy is only needed when backward will be used.
		  ;; Note: The usage of forward below seems a little tricky
		  (forward (,node-name (dtype x))
			   (!copy x)
			   (if *no-grad*
			       y ;; !copy y is always ignored.
			       (!copy y)))))))
  
  (define-arithmetic-node-caller
      !matrix-add
    AddNode
    "adds"
    "and"
    "+")
  (define-arithmetic-node-caller
      !matrix-sub
    SubNode
    "substracts"
    "by"
    "-")
  (define-arithmetic-node-caller
      !matrix-mul
    MulNode
    "multiplies"
    "and"
    "*")
  (define-arithmetic-node-caller
      !matrix-div
    DivNode
    "divides"
    "by"
    "/"))

(with-export !inverse
  (defun !inverse (tensor)
    "## [function] !inverse

```lisp
(!inverse tensor)
```

The function `!inverse` calls `InverseTensorNode`, and finds the inverse of the received Tensor/Scalar, returning a new tensor.

```math
X_{copy}\\gets{1 / X}
```

### Inputs

tensor[ScalarTensor/AbstractTensor/Number]
"
    (let* ((X (if (numberp tensor)
		  (make-tensor tensor)
		  tensor)))
      (if (scalar-p X)
	  (!sas-div 1 X)
	  (forward (InverseTensorNode (dtype X)) (!copy X))))))

;; update docs
(macrolet ((define-scalar-mat-node-caller (name node-name document)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (scalar x)
		  ,(format nil "
## [function] ~(~a~)

```lisp
(~(~a~) scalar x)
```

The function ~a computes following operation with calling `~a`, returning a new tensor.

```math
~a
```

### Inputs

`scalar` could be one of `ScalarTensor` or `number`

`tensor` `AbstractTensor` (should not be a scalar)
"
			   (symbol-name name)
			   (symbol-name name)
			   (symbol-name name)
			   (symbol-name node-name)
			   document)
		  (forward (,node-name (dtype x))
			   (!copy x)
			   (!copy (number->stensor scalar x)))))))
  (define-scalar-mat-node-caller
      !scalar-add ScalarAdd
    "X_{copy}\\gets{X + scalar}")
  (define-scalar-mat-node-caller
      !scalar-sub ScalarSub
    "X_{copy}\\gets{X - scalar}")
  (define-scalar-mat-node-caller
      !scalar-mul ScalarMul
    "X_{copy}\\gets{X * scalar}")
  (define-scalar-mat-node-caller
      !scalar-div ScalarDiv
    "X_{copy}\\gets{X / scalar}"))

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

(macrolet ((define-sas-op (name node-name op)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y)
		  ,(format nil "
## [function] ~(~a~)

The function ~(~a~) provides differentiable scalar-and-scalar operation.

Calling a node `~a`, the function performs following operation:

```math
x_{copy}\\gets{x ~a y}
```

### Inputs

`x` `y` could be one of: `ScalarTensor` or `number`

"
			   (symbol-name name)
			   (symbol-name name)
			   (symbol-name node-name)
			   op)
		  (forward (,node-name)
			   (!copy (number->stensor x y))  ;; Returns X
			   (!copy (number->stensor y x))) ;; Returns Y
		  ))))
  (define-sas-op !sas-add ScalarAndScalarAdd "+")
  (define-sas-op !sas-sub ScalarAndScalarSub "-")
  (define-sas-op !sas-mul ScalarAndScalarMul "*")
  (define-sas-op !sas-div ScalarAndScalarDiv "/"))

;; ===============================================================
;; Defines general-purpose functions.
;; ===============================================================

(defun scalartensor-p (tensor)
  (scalar-p tensor))

(macrolet ((define-arith-function (name
				   invertor
				   broadcast-op
				   scalar-and-scalar-operation
				   scalar-operation
				   matrix-operation)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y)
		  ,(format nil
			   "
## [function] ~(~a~)

```lisp
(~(~a~) x y)
```

The function provides general-purpose arithmetic operation.

Given type of tensors, this function dispatches these functions automatically:

1. `~(~a~)`

2. `~(~a~)`

3. `~(~a~)`

### Inputs

`x` `y` could be one of `AbstractTensor` `number` `ScalarTensor`

### SideEffects

None
"
			   (symbol-name name)
			   (symbol-name name)
			   (symbol-name scalar-and-scalar-operation)
			   (symbol-name scalar-operation)
			   (symbol-name matrix-operation))
		  (let ((x (number->stensor x y))  ;; Returns X
			(y (number->stensor y x))) ;; Returns Y
		    (cond
		      ((and (scalar-p x)
			    (scalar-p y))
		       (,scalar-and-scalar-operation x y))
		      ((scalar-p x)
		       ;; X is scalar, Y is matrix.
		       ;; (!sub 1.0 (10 10))
		       ;; (!div 1.0 (10 10))
		       ;; Transform ->
		       ;; -> (!add (- (10 10)) 1.0)
		       ;; -> (!mul (/ (10 10)) 1.0)
		       ,(if broadcast-op
			    `(,broadcast-op x (,@invertor y))
			    `(,scalar-operation x y)))
		      ((scalar-p y)
		       ;; (!sub (10 10) 1.0)
		       ;; (!div (10 10) 1.0)
		       (,scalar-operation y x))
		      (T
		       (,matrix-operation x y))))))))
  (define-arith-function
      !add (progn)           nil !sas-add !scalar-add !matrix-add)
  (define-arith-function
      !sub (!scalar-mul -1) !add !sas-sub !scalar-sub !matrix-sub)
  (define-arith-function
      !mul (progn)          nil  !sas-mul !scalar-mul !matrix-mul)
  (define-arith-function
      !div (!inverse)       !mul !sas-div !scalar-div !matrix-div))


;; ===============================================================
;; Destructive Functions Family: A+=B A-=B A*=B A/=B
;; ===============================================================

(macrolet ((define-darith-function (name
				    matrix-operation
				    op)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (A B)
		  ,(format nil "
## [function] ~(~a~)

```
(~(~a~) A B)
```

The function provides destructive operation of `~a` which computes:

```math
A\\gets{A ~a B}
```

### Inputs

`A` `B` both of them are `AbstractTensor` with the same shape, not `ScalarTensor`.

"
			   (symbol-name name)
			   (symbol-name name)
			   (symbol-name matrix-operation)
			   op)
		  (declare (type AbstractTensor A B))
		  (assert (or (not (scalar-p A))
			      (not (scalar-p B)))
			  nil
			  "Assertion Failed with A and B both aren't scalar.")
		  (forward (,matrix-operation (dtype A)) A B)))))
  (define-darith-function A+=B AddNode "+")
  (define-darith-function A-=B SubNode "-")
  (define-darith-function A*=B MulNode "*")
  (define-darith-function A/=B DivNode "/"))

;; ===============================================================
;; Destructive Functions Family: A+=scal A-=scal A*=scal A/=scal
;; ===============================================================

(macrolet ((define-darith-function (name
				    matrix-operation
				    op)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (A scalar)
		  ,(format nil "
## [function] ~(~a~)

```
(~(~a~) A scalar)
```

The function provides destructive scalar-matrix operation of `~a`, which computes:

```math
A\\gets{A ~a scalar}
```

### Inputs

`A` AbstractTensor

`scalar` could be one of `number` or `AbstractTensor`
"
			   (symbol-name name)
			   (symbol-name name)
			   (symbol-name matrix-operation)
			   op)
		  (if (numberp scalar)
		      (forward (,matrix-operation (dtype A)) A (make-tensor scalar))
		      (forward (,matrix-operation (dtype A)) A scalar)))))
	   (define-darith-function1 (name broadcast op op1 arg op-name)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (A scalar)
		  ,(format nil "
## [function] ~(~a~)

```
(~(~a~) A scalar)
```

The function provides destructive scalar-matrix operation which computes:

```math
A\\gets{A ~a scalar}
```

### Inputs

`A` AbstractTensor

`scalar` could be one of `number` or `AbstractTensor`"
			   (symbol-name name)
			   (symbol-name name)
			   op-name)
		  (if (numberp scalar)
		      (,broadcast A (,op  (number->stensor scalar A)))
		      (,broadcast A (,op1 ,arg (number->stensor scalar A))))))))
  (define-darith-function  A+=scal ScalarAdd "+")
  (define-darith-function1 A-=scal A+=scal - !mul -1 "-")
  (define-darith-function  A*=scal ScalarMul "*")
  (define-darith-function1 A/=scal A*=scal / !div 1 "*"))


(macrolet ((define-arith-op (waffe-func lisp-op)
	     `(progn
		(export ',lisp-op)
		(defun ,lisp-op (&rest numbers)
		  ,(format nil "
## [function] ~a

Is the equivalent to just doing `(reduce ~a numbers)`"
			   lisp-op
			   waffe-func)
		  (reduce ,waffe-func numbers)))))
  (define-arith-op #'!add !+)
  (define-arith-op #'!sub !-)
  (define-arith-op #'!mul !*)
  (define-arith-op #'!div !/))


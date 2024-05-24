
(in-package :cl-waffe2/base-impl)

(deftype function-args-t ()
  "Indicates the list of types that can be arguments of functions."
  `(or AbstractTensor number symbol cl-waffe2/vm:LazyAxis))

;;  ~ Utils for inferencing types ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(defun fold-constant-scalar-p (tensor)
  "Returns T if the tensor is independent constant"
  (and
   (equal (tensor-facet tensor) :exist)
   (not (requires-grad tensor))
   (scalar-p tensor)
   (null (tensor-variables tensor))
   (null (tensor-backward  tensor))
   (numberp (cl-waffe2/vm.generic-tensor::vec tensor))))

(defun infer-scalar-type (scalar tensor)
  "This function always returns ScalarTensor, whenever scalar is number or ScalarTensor. tensor[number or AbstractTensor] is used in order to determine the dtype of scalar."
  (if (numberp scalar)
      (make-tensor scalar :dtype (if (numberp tensor)
				     (dtype-of tensor)
				     (if (typep tensor 'AbstractTensor)
					 (dtype tensor)
					 (dtype-of scalar))))
      (if (typep scalar 'AbstractTensor)
	  scalar
	  (make-tensor scalar :dtype :uint32))))

(defun infer-scalar-tensor (scalar tensor &key (rank 1))
  "This function always returns ScalarTensor, whenever scalar is number or ScalarTensor. tensor[number or AbstractTensor] is used in order to determine the dtype of scalar."
  (if (numberp scalar)
      (make-tensor (make-list rank :initial-element 1)
		   :dtype (if (numberp tensor)
			      (dtype-of tensor)
			      (if (typep tensor 'AbstractTensor)
				  (dtype tensor)
				  (dtype-of scalar)))
		   :initial-element scalar)
      (if (typep scalar 'AbstractTensor)
	  (if (scalar-p scalar)
	      (if (fold-constant-scalar-p scalar)
		  (make-tensor (make-list rank :initial-element 1)
			       :dtype (dtype scalar)
			       :initial-element (tensor-vec scalar))
		  (!rankup scalar (1- rank)))
	      scalar)
	  (make-tensor (make-list rank :initial-element 1)
		       :dtype :uint32
		       :initial-element scalar))))
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
			  :save-for-backward ,sv4bw
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
	    (!mul dy dy))))))

;; ===============================================================
;; Defun Parts
;; ===============================================================

(declaim (ftype (function (AbstractTensor AbstractTensor &key (:in-place boolean)) (values AbstractTensor &optional))
		!matrix-add
		!matrix-sub
		!matrix-mul
		!matrix-div))
(macrolet ((define-arithmetic-node-caller (name node-name ops prep f)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y &key (in-place nil))
		  (declare (ftype (function (AbstractNode AbstractTensor AbstractTensor) (values AbstractTensor &optional))
				  forward))
		  ,(format nil "
## [function] ~(~a~)

```lisp
(~(~a~) x y &key (in-place nil))
```

The function `~(~a~)` calls `~a` and ~a X ~a Y element-wise, returning a new tensor.

```math
X_{copy}\\gets{X ~a Y}
```

### Inputs

`X` and `Y` must be a AbstractTensor (not a ScalarTensor), with the same shape.

`in-place` set T to make it in-place

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
		  (forward (,node-name (dtype x))
			   (if in-place x (!copy x))			       
			   y)))))
  
  (define-arithmetic-node-caller
      !matrix-add
    AddNode
    "adds"
    "and"
    "+")
  (define-arithmetic-node-caller
      !matrix-sub
    SubNode
    "subtracts"
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

;; TODO: Replace with !reciprocal
(declaim (ftype (function (function-args-t) (values AbstractTensor &optional)) !reciprocal))
(with-export !reciprocal
  (defun !reciprocal (tensor)
    "## [function] !reciprocal

```lisp
(!reciprocal tensor)
```

Finds the reciprocal of tensor.

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
	  (!div
	   (->contiguous
	    (!view
	     (infer-scalar-tensor 1 x :rank (dims x))
	     (broadcast-to x)))
	   X)))))

(declaim (ftype (function (function-args-t AbstractTensor &key (:in-place boolean)) (values AbstractTensor &optional))
		!scalar-add
		!scalar-sub
		!scalar-mul
		!scalar-div))
(macrolet ((define-scalar-mat-node-caller (name node-name document)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (scalar x &key (in-place nil))
		  ,(format nil "
## [function] ~(~a~)

```lisp
(~(~a~) scalar x &key (in-place nil))
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
			   (if in-place
			       x
			       (!copy x))
			   (!view
			    (infer-scalar-tensor scalar x :rank (dims x))
			    (broadcast-to x)))))))
  (define-scalar-mat-node-caller
      !scalar-add AddNode
    "X_{copy}\\gets{X + scalar}")
  (define-scalar-mat-node-caller
      !scalar-sub SubNode
    "X_{copy}\\gets{X - scalar}")
  (define-scalar-mat-node-caller
      !scalar-mul MulNode
    "X_{copy}\\gets{X * scalar}")
  (define-scalar-mat-node-caller
      !scalar-div DivNode
    "X_{copy}\\gets{X / scalar}"))

;; ===============================================================
;; Scalar-And-Scalar Defnode And Functions.
;; ===============================================================

(macrolet ((define-sas-node (name lisp-op sv4bw backward)
	     `(progn
		(defnode (,name (myself)
			  :out-scalar-p t
			  :save-for-backward ,sv4bw
			  :where (A[scal] B[scal] -> A[scal] where scal = 1)
			  :backward ,backward))
		(define-impl (,name :device ScalarTensor)
			     :forward ((self x y)
				       (let ((t1 (dtype->lisp-type (dtype x)))
					     (t2 (dtype->lisp-type (dtype y))))
					 `(and
					   (setf (the ,t1 (tensor-vec ,x))
						 (,',lisp-op (the ,t1 (tensor-vec ,x))
							     (the ,t2 (tensor-vec ,y))))
					   ,x)))))))
  (define-sas-node ScalarAndScalarAdd +
    (nil nil)
    ((self dout x y)
     (declare (ignore x y))
     (values dout dout)))
  (define-sas-node ScalarAndScalarSub -
    (nil nil)
    ((self dout x y)
     (declare (ignore x y))
     (values dout (!sas-mul -1 dout))))
  (define-sas-node ScalarAndScalarMul *
    (t t)
    ((self dout x y)
     (values (!sas-mul y dout)
	     (!sas-mul x dout))))
  (define-sas-node ScalarAndScalarDiv /
    (t t)
    ((self dout dx dy)
     ;; ∂/∂x = 1/y
     ;; ∂/∂y = -x/y^2
     (values (!sas-div dout dy)
	     (!sas-div
	      (!sas-mul dx (!sas-mul -1 dout))
	      (!mul dy dy))))))

(declaim (ftype (function (function-args-t function-args-t &key (:in-place boolean)) (values AbstractTensor &optional))
		!sas-add !sas-sub !sas-mul !sas-div))
(macrolet ((define-sas-op (name node-name op lisp-op)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y &key (in-place nil))
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
		  (let ((x (infer-scalar-type x y))
			(y (infer-scalar-type y x)))
		    (when (and (fold-constant-scalar-p x)
			       (fold-constant-scalar-p y))
		      (return-from ,name
			(make-tensor
			 (,lisp-op (tensor-vec x) (tensor-vec y))
			 :dtype (dtype x)
			 :order (order x))))
		    
		    (forward (,node-name)
			     (if in-place
				 x
				 (!copy x))  ;; Returns X
			     y)))))) ;; Returns Y
  (define-sas-op !sas-add ScalarAndScalarAdd "+" +)
  (define-sas-op !sas-sub ScalarAndScalarSub "-" -)
  (define-sas-op !sas-mul ScalarAndScalarMul "*" *)
  (define-sas-op !sas-div ScalarAndScalarDiv "/" /))

;; ===============================================================
;; Defines general-purpose functions.
;; ===============================================================

(defun scalartensor-p (tensor)
  (if (typep tensor 'AbstractTensor)
      (scalar-p tensor)
      t))

(declaim (ftype (function (function-args-t function-args-t &key (:in-place boolean)) (values AbstractTensor &optional))
		!add !sub !mul !div))
(macrolet ((define-arith-function (name
				   invertor
				   broadcast-op
				   scalar-and-scalar-operation
				   scalar-operation
				   matrix-operation)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (x y &key (in-place t))
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
		  (cond
		    ((and (scalartensor-p x)
			  (scalartensor-p y))
		     (,scalar-and-scalar-operation x y :in-place in-place))
		    ((scalartensor-p x)
		     ;; X is scalar, Y is matrix.
		     ;; (!sub 1.0 (10 10))
		     ;; (!div 1.0 (10 10))
		     ;; Transform ->
		     ;; -> (!add (- (10 10)) 1.0)
		     ;; -> (!mul (/ (10 10)) 1.0)
		     ,(if broadcast-op
			  `(,broadcast-op x (,@invertor y) :in-place in-place)
			  `(,scalar-operation x y :in-place in-place)))
		    ((scalartensor-p y)
		     ;; (!sub (10 10) 1.0)
		     ;; (!div (10 10) 1.0)
		     (,scalar-operation y x :in-place in-place))
		    (T
		     (,matrix-operation x y :in-place in-place)))))))
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

(declaim (ftype (function (AbstractTensor AbstractTensor) (values AbstractTensor &optional))
		A+=B
		A-=B
		A*=B
		A/=B))
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
		  (,matrix-operation A B :in-place t)))))
  (define-darith-function A+=B !add "+")
  (define-darith-function A-=B !sub "-")
  (define-darith-function A*=B !mul "*")
  (define-darith-function A/=B !div "/"))

;; ===============================================================
;; Destructive Functions Family: A+=scal A-=scal A*=scal A/=scal
;; ===============================================================

(declaim (ftype (function (AbstractTensor function-args-t) (values AbstractTensor &optional))
		A+=scal
		A-=scal
		A*=scal
		A/=scal))
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
		  (,matrix-operation A scalar :in-place t)))))
  (define-darith-function A+=scal !add "+")
  (define-darith-function A-=scal !sub "-")
  (define-darith-function A*=scal !mul "*")
  (define-darith-function A/=scal !div "/"))

(declaim (ftype (function (&rest function-args-t) (values AbstractTensor &optional))
		!+ !- !* !/))
(macrolet ((define-arith-op (waffe-func lisp-op)
	     `(progn
		(export ',lisp-op)
		(defun ,lisp-op (&rest numbers)
		  ,(format nil "
## [function] ~a

Is the equivalent to just doing `(reduce ~a numbers)`

### Example

```
(~a 1 2 3 4 5)
```"
			   lisp-op
			   waffe-func
			   waffe-func)
		  (reduce ,waffe-func numbers)))))
  (define-arith-op #'!add !+)
  (define-arith-op #'!sub !-)
  (define-arith-op #'!mul !*)
  (define-arith-op #'!div !/))


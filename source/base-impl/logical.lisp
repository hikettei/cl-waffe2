
(in-package :cl-waffe2/base-impl)

;; =============================================
;; Defnode Parts
;; =============================================


(export '(logical-condition logical-true-then logical-false-then
	  Where-Operation-Node Compare-Operation-Node))

(defnode (Where-Operation-Node (myself condition true-then false-then &key (compiler-info nil))
	  ;;:no-grad t
	  :where (A[~] OUT[~] -> OUT[~])
	  :slots ((condition :initarg :condition :type function :reader logical-condition)
		  (compiler-info :initarg :compiler-info :type list :reader logical-compiler-info)
		  (true-then :initarg :true-then :reader logical-true-then)
		  (false-then :initarg :false-then :reader logical-false-then))
	  :backward ((self dout da do)
		     (declare (ignore dout da do))
		     (values nil nil))
	  :documentation "Where-Operation-Node is a node which set `true-then`, if the result of calling `condition` with each element of A, is t and if it is NIL, set `false-then` at corresponding position.

### Constructor

```
(Where-Operation-Node condition true-then false-then)
```

`true-then` and `false-then` is a number.

`condition` a single argument function, each element of A is argument. (e.g.: this could be `#'evenp` `#'oddp` etc...)
"))

(defnode (Compare-Operation-Node (myself condition true-then false-then)
	  ;;:no-grad t
	  :where (A[~] B[~] OUT[~] -> OUT[~])
	  :slots ((condition :initarg :condition :type function :reader logical-condition)
		  (true-then :initarg :true-then :reader logical-true-then)
		  (false-then :initarg :false-then :reader logical-false-then))
	  :backward ((self dout da db do)
		     (declare (ignore dout da db do))
		     (values nil nil nil))
	  :documentation "Compare-Operation-Node is a node which set `true-then`, if the result of calling `condition` with each element of A and B, if it is NIl set `false-then` at corresponding position.

### Constructor

```
(Compare-Operation-Node condition true-then false-then)
```

`true-then` and `false-then` is a number.

`condition` a two arguments function, each element of A and B is argument. (e.g.: this could be `#'>` or `#'<` etc...)
"))


;; =============================================
;; Fundamental APIs
;; =============================================

;; how to determine dtype
;; can compare being optimized?
(export '!where)
(defun !where (tensor condition &key (true-then 1) (false-then 0) (out nil) (compiler-info nil))
  "
## [function] !where

```
(!where tensor condition &key (true-then 1) (false-then 0) (out nil))
```

The function !where returns a elements selected-from `true-then` or `false-then`, depending on condition.

The operation is defined as:

```math
\\begin{equation}
  out_i=
  \\begin{cases}
    \\text{true-then} & condition(X_i) \\\\
    \\text{false-then} & \\text{otherwise}
  \\end{cases}
\\end{equation}
```

(where X = tensor)

### Inputs

`out` place to set the result
`condition` an funcallable function. (e.g.: #'evenp #'oddp etc...)
"
  (declare (type AbstractTensor tensor)
	   (type function condition)
	   (type number true-then false-then))

  (assert (not (scalar-p tensor))
	  nil
	  "!where: Assertion Failed with the given tensor isn't ScalarTensor.")

  (let ((out (or out (make-input (shape tensor) nil
				 :order (order tensor)
				 :dtype (dtype tensor))))
	(true-then  (coerce true-then (dtype->lisp-type (dtype tensor))))
	(false-then (coerce false-then (dtype->lisp-type (dtype tensor)))))
    (forward (Where-Operation-Node condition true-then false-then :compiler-info compiler-info)
	     tensor
	     out)))

(export '!compare)
(defun !compare (tensor1 tensor2 condition &key (true-then 1) (false-then 0) (out nil))
    "
## [function] !where

```
(!compare tensor1 tensor2 condition &key (true-then 1) (false-then 0) (out nil))
```

The function !compare returns a elements selected-from `true-then` or `false-then`, depending on condition.

The operation is defined as:

```math
\\begin{equation}
  out_i=
  \\begin{cases}
    \\text{true-then} & condition(X_i, Y_i) \\\\
    \\text{false-then} & \\text{otherwise}
  \\end{cases}
\\end{equation}
```

(where X = tensor1, Y=tensor2)

### Inputs

`out` place to set the result
`condition` an funcallable function. (e.g.: #'> #'< etc...)"
  (declare (type AbstractTensor tensor1 tensor2)
	   (type function condition)
	   (type number true-then false-then))

  (assert (and (not (scalar-p tensor1)) (not (scalar-p tensor2)))
	  nil
	  "!compare: Assertion Failed because the given tensor1, and tensor2 should be a matrix, not a scalar.")

  (let ((out (or out (make-input (shape tensor1) nil
				 :order (order tensor1)
				 :dtype (dtype tensor1))))
	(true-then  (coerce true-then  (dtype->lisp-type (dtype tensor1))))
	(false-then (coerce false-then (dtype->lisp-type (dtype tensor1)))))
    (forward (Compare-Operation-Node condition true-then false-then)
	     tensor1
	     tensor2
	     out)))

;; =============================================
;; More APIs
;; =============================================

(macrolet ((define-cmp-scal-operation (name operator)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (A scal &key (out nil) (true-then 1) (false-then 0))
		  ,(format nil "
## [function] ~(~a~)

```
(~(~a~) A scal &key (out nil) (true-then 1) (false-then 0))
```

The function ~(~a~) sets `true-then` if the equation: `element ~a scal` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` AbstractTensor
`scal` number (not a ScalarTensor)

(TODO: ScalarTensor as a `scal` argument)"
			   name
			   name
			   name
			   operator)
		  (!where A #'(lambda (x) (,operator x scal))
			  :compiler-info (list #',operator scal)
			  :true-then  true-then
			  :false-then false-then
			  :out out)))))
  (define-cmp-scal-operation A>scal >)
  (define-cmp-scal-operation A<scal <)
  (define-cmp-scal-operation A>=scal >=)
  (define-cmp-scal-operation A<=scal <=)
  (define-cmp-scal-operation A=scal =))

(macrolet ((define-cmp-mat-operation (name operator)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (A B &key (out nil) (true-then 1) (false-then 0))
		  ,(format nil "
## [function] ~(~a~)

```
(~(~a~) A B &key (out nil) (true-then 1) (false-then 0))
```

The function ~(~a~) sets `true-then` if the equation: `A ~a B` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` `B` AbstractTensor to be compared.
"
			   name
			   name
			   name
			   operator)
		  (!compare A B #',operator
			    :true-then  true-then
			    :false-then false-then
			    :out out)))))
  (define-cmp-mat-operation A>B >)
  (define-cmp-mat-operation A<B <)
  (define-cmp-mat-operation A>=B >=)
  (define-cmp-mat-operation A<=B <=)
  (define-cmp-mat-operation A=B =))
		       


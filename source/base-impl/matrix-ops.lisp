
(in-package :cl-waffe2/base-impl)

;; Implement: Matmul/Dot/ArgMax/ArgMin

(defnode (MatMulNode (myself dtype &key transpose-a transpose-b)
	  :where (A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :slots ((transpose-a :initarg :transpose-a :type boolean :reader trans-a?)
		  (transpose-b :initarg :transpose-b :type boolean :reader trans-b?))
	  :backward ((self dout da db do)
		     (declare (ignore do))
		     (values
		      (!matmul dout (!t db))
		      (!matmul (!t da) dout)
		      nil))
	  :documentation "MatmulNode Computes a matrix multiplication of given A and B, set the result to C.

```math
C\\gets{gemm(1.0, A, B, 0.0, C)}
```

### Constructor

```
(MatMulNode dtype &key transpose-a transpose-b)
```

`dtype` dtype to use.

`transpose-a` `transpose-b` set t to call with transposing (reversing the last two axes the matrix).

"))

(defnode (LazyTransposeNode (self)
	  :where (A[~ i j] -> A[~ j i])
	  :slots ((raw-tensor :accessor raw-tensor))
	  :documentation "LazyTransposeNode is the matmul-dedicated node which supplies the lazy-transpose feature.

Internally, This Node Returns The Given A itself but taking transpose of A's shape.

If the computation node is like: [LazyTransposeNode] -> [MatmulNode], then transpose will be done with NO overhead."
	  :backward ((self dout dx)
		     (declare (ignore dx))
		     (values (!t dout)))))

(define-impl (LazyTransposeNode :device t)
	     :forward ((self x) (setf (raw-tensor self) x) `(progn ,x)))

(defun read-untransposed (tensor)
  ""
  (if (transposed-p tensor)
      (raw-tensor (tensor-backward tensor))
      tensor))

(defun transposed-p (tensor)
  "Return T if previous-node is LazyTransposeNode"
  (subtypep (class-of (tensor-backward tensor)) 'LazyTransposeNode))


;; :== The problem is that ==============:
;;  !flexible(!t(x)).is_transposed? = NIL
;;  !t(!flexible(x)).is_flexible?   = T
;; :=====================================:
(defun !t (tensor)
  "
## [function] !t

```
(!t tensor)
```

Applies Lazy-Transpose to the given tensor.

The function is matmul-dedicated, so cooperationg with other operations (e.g.: !add) will cause the wrong result. (Internally, it is the equivalent to calling `!reshape`)

### Current Problem

Inconsistency of operations:

```lisp
!flexible(!t(x)).is_transposed? = NIL
!t(!flexible(x)).is_flexible?   = T
```
"
  ;; extend flexible?
  (extend-states (forward (LazyTransposeNode) tensor) tensor))

(defun !matmul (x y
		&key
		  (out nil)
		  (transpose-x nil)
		  (transpose-y nil)
		&aux
		  (x (if transpose-x (!t x) x))
		  (y (if transpose-y (!t y) y))
		  (transpose-x (transposed-p x))
		  (transpose-y (transposed-p y)))
  "
## [function] !matmul

```lisp
(!matmul x y &key (out nil) (transpose-x nil) (transpose-y nil))
```

Computing a matrix multiplication of X and Y, the function set the result into out.

```math
out\\gets{gemm(1.0, x, y, 0.0, out)}
```

### Inputs

`transpose-x` `transpose-y` If t, the tensor is called with `(!t tensor)`

### Lazy-Transpose

Call the function `(!t tensor)` in advance to transpose the tensor without overheads.

```
(!matmul (!t (randn `(5 3))) (randn `(5 3)))
```
"
  (let* ((i  (nth 0 (last (shape x) 2)))
	 (jx (nth 1 (last (shape x) 2)))
	 (jy (nth 0 (last (shape y) 2)))
	 (k  (nth 1 (last (shape y) 2)))
	 ;; the way to make out's shape
	 (larger-shape (if (> (length (shape x)) (length (shape y)))
			   (shape x)
			   (shape y)))
	 ;; the longer dim's shape is adapted.
	 (out (or out (make-input `(,@(butlast larger-shape 2) ,i ,k) nil
				  :dtype (dtype x)
				  :order (order x)))))

    (when (not (shape-equal jx jy))
      (shaping-error
       "!matmul failed because the last two shapes of the two given matrices are invaild.
The operation is: A[~~ i j] B[~~ j k] C[~~ i k] -> C[~~ i k]
                        ^      ^
                     j doesn't match: ~a and ~a
Shapes: A = ~a, B = ~a"
       jx
       jy
       (shape x)
       (shape y)))
    
    (forward (MatmulNode (dtype x)
	      :transpose-a transpose-x
	      :transpose-b transpose-y)
	      x
	      y
	      out))) ;; !flexible

(with-export !dot
  (defun !dot (x y)
    "
## [function] !dot

```
(!dot x y)
```

Finds a dot product of x and y. Unlike `numpy.dot`, `!dot` intentionally only supports computing the dot product of two 1D tensors with the same number of elements.

```lisp
(proceed (!dot (randn `(100)) (randn `(10 10))))
{CPUTENSOR[float] :shape (1) -> :view (<0>) -> :visible-shape (1) :named ChainTMP115880 
  :vec-state [computed]
  (21.594929)
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```"
    (!sum (!mul (!flatten x) (!flatten y)))))

;; (defun einsum)

(export '(ArgMax-Node ArgMin-Node !argmax !argmin))
(defnode (ArgMax-Node (myself out-size)
	  :where (A[~] OUT[out-size] -> OUT[out-size])
	  :backward ((self dout da do)
		     (declare (ignore dout da do))
		     (values nil nil))
	  :documentation "ArgMax-Node finds a maximum value of all elements in A. `OUT` is overwritten with the result.

A is a target to find a maximum value, and OUT is a place to set the index.

### Constructor

```
(ArgMax-Node out-size)
```

`out-size` the reducted shape of `out`.
")
  (setf (ignore-shape-error myself) t))

(defnode (ArgMin-Node (myself out-size)
	  :where (A[~] OUT[out-size] -> OUT[out-size])
	  :backward ((self dout da do)
		     (declare (ignore dout da do))
		     (values nil nil))
	  :documentation "ArgMin-Node finds a minimum value of all elements in A. `OUT` is overwritten with the result.

A is a target to find a minimum value, and OUT is a place to set the index.

### Constructor

```
(ArgMin-Node out-size)
```

`out-size` the reducted shape of `out`.")
  (setf (ignore-shape-error myself) t))


(defun !argmax (tensor &key (axis -1) (out nil))
  "
## [function] !argmax

```
(!argmax tensor &key (axis -1) (out nil))
```

The function !argmax computes the indices of maximum values of all elements below the **axis** dimension in the given tensor.

### Inputs

`tensor`

`axis`

`out`

### Returns

AbstractTensor[uint32] with dimensions behind `axis` is replaced with 1."
  (declare (type AbstractTensor tensor)
	   (type fixnum axis))
  (let* ((axis (if (< axis 0)
		   (+ (length (shape tensor)) axis)
		   axis))
	 (out-shape (butlast (shape tensor) axis))
	 (x   (apply #'!reshape tensor `(,@out-shape t)))
	 (out (or out (make-input `(,@out-shape 1) nil
				  :dtype :uint32
				  :order (order tensor)))))
    (forward (ArgMax-Node (shape out)) x out)))

(defun !argmin (tensor &key (axis -1) (out nil))
  "## [function] !argmin

```
(!argmin tensor &key (axis -1) (out nil))
```

The function !argmin computes the indices of minimum values of all elements below the **axis** dimension in the given tensor.

### Inputs

`tensor`

`axis`

`out`

### Returns

AbstractTensor[uint32] with dimensions behind `axis` is replaced with 1."
  (declare (type AbstractTensor tensor)
	   (type fixnum axis))
  (let* ((axis (if (< axis 0)
		   (+ (length (shape tensor)) axis)
		   axis))
	 (out-shape (butlast (shape tensor) axis))
	 (x   (apply #'!reshape tensor `(,@out-shape t)))
	 (out (or out (make-input `(,@out-shape 1) nil
				  :dtype :uint32
				  :order (order tensor)))))
    (forward (ArgMin-Node (shape out)) x out)))

;; (defun !argmax)
;; (defun !argmin)
;; (defun !max)
;; (defun !min)

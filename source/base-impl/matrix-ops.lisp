
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

`transpose-a transpose-b[boolean]` becomes t if the given `a` or `b` needs to be transposed respectively. call `(read-untransposed tensor)` to read untransposed tensor.

"))

(defnode (LazyTransposeNode (self)
	  :where (A[~ i j] -> A[~ i j])
	  :slots ((raw-tensor :accessor raw-tensor))
	  :documentation "LazyTransposeNode is a matmul-dedicated node to implement zero-cost transpose.

The node stores untransposed tensor at `raw-tensor`, when expanding matmul form, you can read it if needed."
	  :backward ((self dout dx)
		     (declare (ignore dx))
		     (values dout))))

(define-impl (LazyTransposeNode :device t)
	     :forward ((self x)
		       (setf (raw-tensor self) x)
		       `(progn (tensor-vec ,x) ,x)))

(defun read-untransposed (tensor)
  ""
  (if (transposed-p tensor)
      (car (tensor-variables (raw-tensor (tensor-backward tensor))))
      tensor))


(defun transposed-p (tensor)
  "Return T if previous-node is LazyTransposeNode"
  (subtypep (class-of (tensor-backward tensor)) 'LazyTransposeNode))


(defun !t (tensor)
  "
## [function] !t

```
(!t tensor)
```

Transposes the last two axes of the given tensor.

When called with !matmul, the operation is ignored.
"
  
  (extend-states (!permute tensor :~ 0 1) tensor))

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

Computing a matrix multiplication of X and Y. The result is stored in out if specified, otherwise creates a new tensor.

```math
out\\gets{gemm(1.0, x, y, 0.0, out)}
```

### Inputs

`transpose-x, transpose-y[boolean]` If t, the inputs are wrapped with `(!t tensor)`.

### Tips: Lazy-Transpose-Node

If the last backward of given arguments are `LazyTransposeNode` (created with the function `!t`), the function `!matmul` will transpose them without making a copy (i.e.: zero-cost transpose). In any other case (the last two dimensions' permution, or view are too complicated), `!matmul` will produce an additional copy for fast computing.

"
  (declare (type AbstractTensor x y)
	   (type (or null AbstractTensor) x y))
  (let* ((i  (nth 0 (last (shape x) 2)))
	 (jx (nth 1 (last (shape x) 2)))
	 (jy (nth 0 (last (shape y) 2)))
	 (k  (nth 1 (last (shape y) 2)))
	 ;; the way to make out's shape
	 ;; The rank is straighten up with larger one.
	 (larger-shape (if (> (length (shape x)) (length (shape y)))
			   (shape x)
			   (shape y)))
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

    (flet ((adjust-layout (tensor transposed?)
	     "Return (values Adjusted-Tensor KeepTranspose?)"
	     (let ((last-two-view (last (tensor-view tensor) 2))
		   (last-two-permute (last (cl-waffe2/vm.generic-tensor::tensor-permute-order tensor) 2))

		   (excepted-permute (last (reverse (loop for i upfrom 0 below (dims tensor) collect i)) 2)))

	       (cond
		 ;; transposed? True when last backward is LazyTranspsoe
		 ;; when LazyTranspos-able <-> Last two axes are subject to swapped.
		 (transposed?
		  (values tensor transposed?))
		 ;; Last two permute is also regarded as LazyTranspose
		 ((and (every
			#'(lambda (x) (eql (force-list x) t))
			last-two-view)
		       (every
			#'=
			last-two-permute
			excepted-permute))
		  ;; Memory-layout is OK. (because last two axes are not polluted)
		  ;; (values tensor nil) <=>
		  (values tensor transposed?))
		 (T
		  ;; Apply Transpose/Permute/View
		  ;; TODO: Delete this copy...
		  (values (->contiguous tensor) nil))))))

      (multiple-value-bind (x x-transpose?) (adjust-layout x transpose-x)
	(multiple-value-bind (y y-transpose?) (adjust-layout y transpose-y)
	  (forward (MatMulNode (dtype x)
			       :transpose-a x-transpose?
			       :transpose-b y-transpose?)
		   x
		   y
		   (adjust-layout out nil)))))))

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

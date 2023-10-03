
(in-package :cl-waffe2/base-impl)


(defun padding (tensor pad-width
		&key
		  (pad-maker #'cl-waffe2/distributions:ax+b)
		  (initargs `(0 0)))
  "
## [function] padding

```lisp
(padding tensor pad-width &key (pad-maker #'ax+b) (initargs `(0 0)))
```

Creating a new InputTensor with shape after padding, the function `padding` moves the given tensor into a new area.

### Implementation

```lisp

(padding (ax+b `(1 3) 0 1) `((1 1) (1 1)))

[Corresponds with...]

                    00000
+++ -> [padding] -> 0+++0
                    00000
```

The operation is performed in the following steps:

First, creates a new tensor with the shape of after padded which is initialized via the `pad-maker` function, where `pad-maker` is an initializer function, that is, functions defined by `define-initializer-function` or exported from the `:cl-waffe2/distribution` package.


```lisp
+++++
+++++
+++++
```

The function `padding` uses the form below to initialize tensors.

```lisp
(apply pad-maker initargs)
```

In default, `(apply #'ax+b `(0 0))`.

Second, makes a view of the new tensor and match its shape to the base tensor. the argument `pad-width` is used to determine offsets of each axis. `pad-width` is the number of values to the edges of each axis. and given as: `((before_1 after_1) (before_2 after_2) ...)`. `0~before_n` and `before_n~last` are the subject to be padded.

```lisp
+++++              -----
+++++ -> [view] -> -+++-
+++++              -----
                       ^ after_2
+ ... visible area
- ... hide area by view
```

Finally, moves all elements in the base tensor into viewed tensor, and later reset the view.

### Inputs

`tensor[AbstractTensor]` tensor to be padded.

`pad-width[list]` the number of the edges and given as: `((before_1 after_1) (before_2 after_2) ...)`. the forward is the same as [np.pad](https://numpy.org/doc/stable/reference/generated/numpy.pad.html). Set t instead of `(before_n after_n)` and ignores the corresponding position of axis.
 
`pad-maker[function]` an initializer-function

`initargs[list]` a list of arguments for `pad-maker`

Note that: the axes to be padded, must be fixnum. not a symbol.

If the shapes does not change before/after padding, returns the given tensor as it is.

### Example

```lisp
(proceed (padding (ax+b `(3 3) 0 1) `((1 1) (1 1))))

{CPUTENSOR[float] :shape (5 5) -> :view (<T> <T>) -> :visible-shape (5 5) :named ChainTMP1104579 
  :vec-state [computed]
  ((0.0 0.0 0.0 0.0 0.0)           
   (0.0 1.0 1.0 1.0 0.0)   
        ...
   (0.0 1.0 1.0 1.0 0.0)
   (0.0 0.0 0.0 0.0 0.0))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```
"
  (assert (= (length pad-width) (dims tensor))
	  nil
	  "padding: Assertion Failed because the rank of tensor and the length of pad-width does not match.")

  (let ((padded-sizes (loop for width in pad-width
			    for shape in (shape tensor)
			    do (assert (or (eql width t) (= (length width) 2))
				       nil
				       "padding: Assertion Failed because pad-width should be given as: `((before_n after_n) ...) but got: ~a`" width)
			    if (eql width t)
			      collect shape
			    else
			      collect (+ (car width) (second width) shape)))
	(padding-view (loop for width in pad-width
			    for shape in (shape tensor)
			    if (eql width t)
			      collect t
			    else
			      collect `(,(car width) ,(1+ (- (+ (car width) shape) (second width)))))))

    ;; If there's no padding, return tensor.
    (when (equal padded-sizes (shape tensor))
      (return-from padding tensor))
    
    (let ((new-tensor (apply pad-maker padded-sizes
			     `(,@initargs
			       :dtype ,(dtype tensor)
			       :order ,(order tensor)))))
      (multiple-value-bind (new-tensor* reverser) (apply #'!view new-tensor padding-view)
	(apply #'!view (!move new-tensor* tensor :force t) reverser)))))



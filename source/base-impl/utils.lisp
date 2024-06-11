
(in-package :cl-waffe2/base-impl)

(defun padding (tensor pad-width
		&key
		  (pad-maker #'make-input)
		  (initargs `(nil)))
  "
## [function] padding

```lisp
(padding tensor pad-width &key (pad-maker #'make-input) (initargs `(nil)))
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

  (let* ((padded-sizes (loop for width in pad-width
			     for shape in (shape tensor)
			     do (assert (or (eql width t) (= (length width) 2))
					nil
					"padding: Assertion Failed because pad-width should be given as: `((before_n after_n) ...) but got: ~a`" width)
			     if (eql width t)
			       collect shape
			     else
			       collect (wf/vm:make-lazyaxis `(+ ,(car width) ,(second width) ,shape))))
	 (padding-view (loop for width in pad-width
			     for shape in (shape tensor)
			     if (eql width t)
			       collect t
			     else
			       collect `(,(car width) ,(wf/vm:make-lazyaxis `(+ ,(car width) ,shape))))))

    ;; If there's no padding, return the tensor as it is.
    (when (equal padded-sizes (shape tensor))
      (return-from padding tensor))
    
    (let ((new-tensor (apply pad-maker padded-sizes
			     `(,@initargs
			       :dtype ,(dtype tensor)
			       :order ,(order tensor)))))
      (multiple-value-bind (new-tensor* reverser) (apply #'!view new-tensor padding-view)
	(setf (wf/t::tensor-visible-shape new-tensor*) (shape tensor))
	(apply #'!view (!move new-tensor* tensor :force t) reverser)))))

(defun !concatenate (axis &rest tensors)
  (let* ((shape (shape
		 (or
		  (car
		   (loop with m = (apply #'max (map 'list (alexandria:compose #'length #'shape) tensors))
			 for tensor in tensors
			 if (= (length (shape tensor)))
			   collect tensor))
		  (car tensors))))
	 (axis (if (>= axis 0)
		   axis
		   (+ 1 (length shape) axis)))
	 (concatenated-shape
	   (loop for s in shape
		 for nth upfrom 0
		 if (= nth axis)
		   collect (wf/vm:make-lazyaxis `(+ ,@(map 'list #'(lambda (x) (nth axis (shape x))) tensors)))
		 else
		   collect s))
	 (out (make-input concatenated-shape nil :dtype (dtype (car tensors)))))

    (flet ((nth-view (total size)
	     (loop for s in concatenated-shape
		   for nth upfrom 0
		   if (= nth axis)
		     collect `(,total ,(wf/vm:make-lazyaxis `(+ ,total ,size)))
		   else
		     collect t)))
      (loop with total = 0
	    for tensor in tensors
	    do (setf out (!move (apply #'!view (!view out) (nth-view total (nth axis (shape tensor)))) tensor :force t)
		     total (wf/vm:make-lazyaxis `(+ ,total ,(nth axis (shape tensor))))))
      (apply #'!view out (make-list (length concatenated-shape) :initial-element t)))))

(defun !tile (tensor repeats)
  (when (typep repeats 'AbstractTensor)
    (setf tensor (lazy-values tensor repeats)
          repeats (loop for s in (shape tensor)
			for axis upfrom 0
			collect `(round (vref ,repeats ,axis)))))
  (loop for s in (shape tensor)
	for axis upfrom 0
	do (setf tensor (!repeat tensor axis (nth axis repeats))))
  tensor)

(defun !repeat (tensor dim n)
  (let* ((shape (shape tensor))
	 (repeated-shape
	   (loop for s in shape
		 for nth upfrom 0
		 if (= nth dim)
		   collect (wf/vm:make-lazyaxis `(* ,s ,n))
		 else
		   collect s)))
    (flet ((repeat (list n)
	     (loop repeat (round (/ n (length list)))
		   append list)))
      (lazy-reduce #'(lambda (&rest i) (apply #'values (repeat i (wf/vm:maybe-observe-axis (nth dim repeated-shape))))) tensor :reduce-to (nth dim repeated-shape) :axis dim))))


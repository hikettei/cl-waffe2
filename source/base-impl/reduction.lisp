
(in-package :cl-waffe2/base-impl)

;; Could be optimized until 10x times faster
(defun !sum (tensor &key (axis t) (-> nil) (keepdims nil) (div-by nil))
  "
## [function] !sum

```
(!sum tensor &key (axis t) (-> nil) (keepdims nil))
```

The function !sum return a node which computes the sum of tensor along the given axis.

### Inputs

`tensor`, a tensor to be reducted.

`axis`[t or fixnum or list] the axis to be reducted. (-1, -2... is ok)

`->` [AbstractTensor or nil] the place to set the result. If nil, creates a new tensor.

`dims`[boolean] If t, the axis reducted is broadcasted.

Return:

`->`[AbstractTensor] the result."
  (declare (type AbstractTensor tensor)
	   (type boolean keepdims)
	   (type (or t list fixnum) axis))
  (let* ((tensor (if (scalar-p tensor)
		     (->mat tensor)
		     tensor))
	 (shape (copy-list (shape tensor)))
	 (view-args (make-list (length shape) :initial-element t))
	 (dims  (length shape)))

    ;; Compute Reduction Size.
    ;; Parse -1 -> 1 for example.
    (typecase axis
      (fixnum
       (if (< axis 0)
	   (setq axis (+ axis dims)))

       (setf (nth axis view-args) `(:broadcast ,(nth axis shape)))
       (setf (nth axis shape) 1))
      (list
       (dolist (dim axis)
	 (let ((tgt (if (< dim 0)
			(+ dim dims)
			dim)))
	   (setf (nth tgt view-args) `(:broadcast ,(nth tgt shape)))
	   (setf (nth tgt shape) 1))))
      (T
       (setq view-args (loop for s in shape
			     collect `(:broadcast ,s)))
       (setq shape (make-list dims :initial-element 1))))

    ;; Use Instead: make-input
    (let* ((out (or -> (make-input shape nil 
				    :dtype (dtype tensor)
				    :order (order tensor))))
	   (out (A*=scal out (make-tensor 0 :dtype (dtype tensor)))))
      ;; TODO: A*=scal is nothing but extravagance to fill with 0.0!, replace this op with !fill

      (assert (equal (shape out) shape)
	      nil
	      "!sum: Assertion Failed because the given out's shape is ~a, but excepted: ~a" (shape out) shape)

      ;; Main Parts
      (multiple-value-bind (out* reverser) (apply #'!view out view-args)
	(if keepdims
	    (if div-by
		(let ((out (apply #'!view (A+=B out* tensor) reverser)))
		  (apply #'!view (!div out div-by) view-args))
		(A+=B out* tensor))
	    (if div-by
	        (!div (apply #'!view (A+=B out* tensor) reverser) div-by)
		(apply #'!view (A+=B out* tensor) reverser)))))))

(defun !mean (tensor &key (axis t) (-> nil) (keepdims nil))
  "
## [function] !mean

```
(!mean tensor &key (axis t) (-> nil) (keepdims nil))
```

The function !mean return a node which computes the average of tensor along the given axis.

### Inputs

`tensor`, a tensor to be reducted.

`axis`[t or fixnum or list] the axis to be reducted. (-1, -2... is ok)

`->` [AbstractTensor or nil] the place to set the result. If nil, creates a new tensor.

`keepdims` [boolean] If t, the axis reducted is broadcasted.

### Return

`->`[AbstractTensor] the result."
  (let* (;;(result (!sum tensor :axis axis :-> -> :keepdims keepdims))
	 (dims   (length (shape tensor)))
	 (reducted-elements (make-tensor 1 :dtype (dtype tensor))))
    
    (typecase axis
      (fixnum
       (if (< axis 0)
	   (setq axis (+ axis dims)))

       (setq reducted-elements (make-tensor (nth axis (shape tensor)) :dtype (dtype tensor))))
      (list
       (dolist (dim axis)
	 (let ((tgt (if (< dim 0)
			(+ dim dims)
			dim)))
	   (setq reducted-elements (!mul reducted-elements (nth tgt (shape tensor)))))))
      (T
       (mapc
	#'(lambda (s)
	    (setq reducted-elements (!mul reducted-elements s)))
	(shape tensor))))

    (!sum tensor :axis axis :-> -> :keepdims keepdims :div-by reducted-elements)))


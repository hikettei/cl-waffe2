
(in-package :cl-waffe2/base-impl)

(defun broadcast-to (object-tensor)
  "
## [function] broadcast-to

Returns the subscript of the `!view` that is broadcasting to be the same shape as the `object-tensor`.

For example:

```lisp
;; x                ... ( 3 3 ) Tensor
;; (!sum x :axis 1) ... ( 3 1 ) Tensor

;; broadcast-to will return: (t `(:broadcast 3))
(!mul x (!view (!sum x :axis 1) (broadcast-to x)))
```
"
  (declare (type AbstractTensor object-tensor))
  #'(lambda (tensor)
      (unless (= (dims tensor) (dims object-tensor))
	(error "broadcast-to: Couldn't broadcast together because ranks does not match. ~a and ~a" tensor object-tensor))
      (loop for shape-tbc    in (shape tensor)
	    for shape-object in (shape object-tensor)
	    if (= shape-tbc shape-object)
	      collect t
	    else
	      collect (progn
			(unless (= shape-tbc 1)
			  (error "broadcast-to: Couldn't broadcast two axes, ~a and ~a" (shape tensor) (shape object-tensor)))
			`(:broadcast ,shape-object)))))

(defun rev-last-two ()
  "
## [function] rev-last-two
"

  #'(lambda (tensor)
      `(,@(butlast (cl-waffe2/vm.generic-tensor::tensor-permute-order tensor) 2)
	,@(reverse (last (cl-waffe2/vm.generic-tensor::tensor-permute-order tensor) 2)))))

(defun torch-order (&rest orders)
  "
## [function] torch-order
Translates the given orders into PyTorch's notation

`(!permute a (torch-order 2 1 0))` is the equivalent to `(!permute a 0 1 2)`.
"
  #'(lambda (tensor)
      (let ((dims (1- (dims tensor))))
	(loop for o in orders
	      if (eql o :~)
		collect o
	      else
		collect (- dims o)))))
            

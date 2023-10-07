
(in-package :cl-waffe2/base-impl)

;; Provides im2col/col2im node

(export '(Im2ColNode N C k-h k-w h-out w-out stride-w stride-h img-out img-out-of h h-of w w-of
	  padding-w padding-h
	  dilation-w dilation-h))
(defnode (Im2ColNode (self N C k-h k-w h-out w-out stride-h stride-w padding-h padding-w dilation-h dilation-w img-out)
	  :slots ((N :initarg :N)
		  (C :initarg :C)
		  (k-h :initarg :k-h)
		  (k-w :initarg :k-w)
		  (h-out :initarg :h-out)
		  (w-out :initarg :w-out)
		  (stride-w :initarg :stride-w)
		  (stride-h :initarg :stride-h)
		  (padding-w :initarg :padding-w)
		  (padding-h :initarg :padding-h)
		  (dilation-w :initarg :dilation-w)
		  (dilation-h :initarg :dilation-h)		  
		  (img-out :initarg :img-out :reader img-out-of)
		  (h :accessor h-of)
		  (w :accessor w-of))
	  :documentation "Im2ColNode is `AbstractNode` which implements forward propagation of [nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html).

The node is only executed through the `cl-waffe2/nn:unfold` function, so arguments for constructors are dispatched automatically. In addition, the tensor `X` it receive will be the one after padding has been performed.

### Slots

`N` indicates the number of batch-size

`C` indicates a channel-size

`k-h`, `k-w` represents the size of kernel. height and width respectively.

`h-out` `w-out` is the size of output weight.

`stride-w stride-h` is the number of strides.

`padding-w padding-h dilation-w dilation-h` more parameters.

`img-out[AbstractTensor]` allocated area to set the result, being accessed by `(img-out-of self)` .

All symbols are exported from `cl-waffe2/base-impl` package and `with-slots` is useful to read all slots.

In order to implement device-specific implementation of `Unfold`, do define-impl for both `Im2ColNode` and `Col2ImNode`.
"
	  ;; Backward: Col[N C k-h k-w h-out w-out] -> X[N C H W] Col[N C k-h k-w h-out w-out]
	  :where (X[N C H W] Col[N C k-h k-w h-out w-out] -> Col[N C k-h k-w h-out w-out])
	  :backward ((self dout x col)
		     (declare (ignore col))
		     (setf (h-of self) (nth 2 (shape x))
			   (w-of self) (nth 3 (shape x)))
		     (with-slots ((N N) (C C) (H H) (W W)				       
				  (h-out h-out) (w-out w-out)
				  (k-h k-h) (k-w k-w)
				  (padding-h padding-h) (padding-w padding-w)
				  (dilation-h dilation-h) (dilation-w dilation-w)
				  (stride-h stride-h) (stride-w stride-w))
			 self
		       (values
			(call (Col2ImNode
			       N C k-h k-w
			       h-out w-out
			       stride-h stride-w
			       padding-h padding-w
			       dilation-h dilation-w
			       (img-out-of self)
			       :H H
			       :W W)
			      dout
			      (img-out-of self))
			nil)))))

(export 'Col2ImNode)
(defnode (Col2ImNode (self N C k-h k-w h-out w-out stride-h stride-w padding-h padding-w dilation-h dilation-w img-out &key (h) (w))
	  :slots ((N :initarg :N)
		  (C :initarg :C)
		  (k-h :initarg :k-h)
		  (k-w :initarg :k-w)
		  (h-out :initarg :h-out)
		  (w-out :initarg :w-out)
		  (stride-w :initarg :stride-w)
		  (stride-h :initarg :stride-h)
		  (padding-w :initarg :padding-w)
		  (padding-h :initarg :padding-h)
		  (dilation-w :initarg :dilation-w)
		  (dilation-h :initarg :dilation-h)		  
		  (img-out :initarg :img-out :reader img-out-of)
		  (h :accessor h-of)
		  (w :accessor w-of))
	  :documentation "Col2ImNode is `AbstractNode` which implements backward propagation of [nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html). It has completely the same slots and arguments to `Im2Col`.

See also: `Im2ColNode` documentation for argument descriptions."
	  :where (Col[N C k-h k-w h-out w-out] X[N C H W] -> X[N C H W])))


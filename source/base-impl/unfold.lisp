
(in-package :cl-waffe2/base-impl)

;; Provides im2col/col2im node

(export '(Im2ColNode N C k-h k-w h-out w-out stride-x stride-y img-out img-out-of h h-of w w-of))
(defnode (Im2ColNode (self N C k-h k-w h-out w-out stride-x stride-y img-out)
	  :slots ((N :initarg :N)
		  (C :initarg :C)
		  (k-h :initarg :k-h)
		  (k-w :initarg :k-w)
		  (h-out :initarg :h-out)
		  (w-out :initarg :w-out)
		  (stride-x :initarg :stride-x)
		  (stride-y :initarg :stride-y)
		  (img-out :initarg :img-out :reader img-out-of)
		  (h :accessor h-of)
		  (w :accessor w-of))
	  :documentation "Im2ColNode is `AbstractNode` which implements forward propagation of [nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html).

The node is only executed through the `cl-waffe2/nn:unfold` function, so arguments for constructors are dispatched automatically. In addition, the tensor `X` it receive will be the one after padding has been performed.

`N` indicates the number of batch-size, `C` is a channel-size. `k-h`, `k-w` represents the size of kernel, height and width respectively. `h-out` `w-out` is the size of output. `stride-x` `stride-y` is the number of stride, for the most case, specified by the stride argument in `Pooling2D` or `Conv2D`. `img-out` is AbstractTensor with the shape of `(N C H-in W-in)`, can be read by `img-out-of`. All symbols are exported from `cl-waffe2/base-impl` package.

In order to implement device-specific implementation of `Unfold`, define-impl `Im2ColNode` and `Col2ImNode`.
"
	  ;; Backward: Col[N C k-h k-w h-out w-out] -> X[N C H W] Col[N C k-h k-w h-out w-out]
	  :where (X[N C H W] Col[N C k-h k-w h-out w-out] -> Col[N C k-h k-w h-out w-out])
	  :backward ((self dout x col)
		     (declare (ignore x col))
		     (with-slots ((N N) (C C) (k-h k-h) (k-w k-w) (h-out h-out) (w-out w-out) (stride-x stride-x) (stride-y stride-y)) self
		       (values
			(call (Col2ImNode N C (h-of self) (w-of self) k-h k-w h-out w-out stride-x stride-y (img-out-of self)) dout)
			nil)))))

(export 'Col2ImNode)
(defnode (Col2ImNode (self N C H W k-h k-w h-out w-out stride-x stride-y img-out)
	  :slots ((N :initarg :N)
		  (C :initarg :C)
		  (k-h :initarg :k-h)
		  (k-w :initarg :k-w)
		  (h-out :initarg :h-out)
		  (w-out :initarg :w-out)
		  (stride-x :initarg :stride-x)
		  (stride-y :initarg :stride-y)
		  (img-out :initarg :img-out :reader img-out-of))
	  :documentation "Col2ImNode is `AbstractNode` which implements backward propagation of [nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html).

See also: `Im2ColNode` documentation for argument descriptions."
	  :where (Col[N C k-h k-w h-out w-out] -> X[N C H W])))


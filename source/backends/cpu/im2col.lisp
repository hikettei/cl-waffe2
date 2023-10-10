
(in-package :cl-waffe2/backends.cpu)

(define-impl-op (Im2ColNode :device CPUTensor :reject-p #'simd-extension-p)
		:forward ((self x col)
			  (setf (h-of self) (nth 2 (shape x))
				(w-of self) (nth 3 (shape x)))
			  (with-slots ((N N) (C C) (H H) (W W)				       
				       (h-out h-out) (w-out w-out)
				       (k-h k-h) (k-w k-w)
				       (padding-h padding-h) (padding-w padding-w)
				       (dilation-h dilation-h) (dilation-w dilation-w)
				       (stride-h stride-h) (stride-w stride-w))
			      self
			    (with-tensor-ptrs ((x* x) (col* col))
			      (funcall (make-im2col (dtype x) (order x))
				       col*
				       (car (shape x)) C H W
				       h-out w-out
				       k-h k-w
				       padding-h padding-w
				       stride-h stride-w
				       dilation-h dilation-w
				       x*))
			    col)))

(define-impl-op (Col2ImNode :device CPUTensor :reject-p #'simd-extension-p)
		:forward ((self col output-to)
			  (setf (h-of self) (nth 2 (shape output-to))
				(w-of self) (nth 3 (shape output-to)))
			  (with-slots ((N N) (C C) (H H) (W W)				       
				       (h-out h-out) (w-out w-out)
				       (k-h k-h) (k-w k-w)
				       (padding-h padding-h) (padding-w padding-w)
				       (dilation-h dilation-h) (dilation-w dilation-w)
				       (stride-h stride-h) (stride-w stride-w))
			      self
			    (with-tensor-ptrs ((col* col) (out* output-to))
			      (funcall (make-col2im (dtype col) (order col))
				       col*
				       (car (shape col)) C H W
				       h-out w-out
				       k-h k-w
				       padding-h  padding-w
				       stride-h   stride-w
				       dilation-h dilation-w
				       out*))
			    output-to)))



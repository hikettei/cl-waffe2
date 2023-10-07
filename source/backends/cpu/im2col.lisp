
(in-package :cl-waffe2/backends.cpu)

;; Git add
;; multiple devices
;; im2col
;; col2im


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
				       N C H W
				       h-out w-out
				       k-h k-w
				       padding-h padding-w
				       stride-h stride-w
				       dilation-h dilation-w
				       x*))
			    col)))


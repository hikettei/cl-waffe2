
(in-package :cl-waffe2/nn)

;;
;; Memo/References:
;;
;; MEC: Memory-efficient Convolution for Deep Neural Network
;; https://arxiv.org/pdf/1706.06873.pdf
;; https://github.com/oneapi-src/oneDNN/blob/master/src/cpu/gemm_convolution_utils.cpp
;; https://blog.cybozu.io/entry/xbyak_for_fugaku
;; https://github.com/melisgl/mgl-mat/blob/94a6351f259fa6d0e0c34e266f72e0cd6bf9881f/src/convolve.lisp#L3
;; https://www10.cs.fau.de/publications/theses/2022/Master_HolzmannMichael.pdf
;;


(define-with-typevar
    (im2col-caller u) (padded-x col N C filter-h filter-w out-h out-w stride-x stride-y)
  (declare (optimize (speed 3))
           (type AbstractTensor padded-x col)
	   (type (unsigned-byte 32) N C filter-h filter-w out-h out-w stride-x stride-y))
  (let* ((strides   (tensor-stride col))
	 ;; strides on col
	 (n-stride  (nth 0 strides))
	 (c-stride  (nth 1 strides))
	 (fh-stride (nth 2 strides))
	 (fw-stride (nth 3 strides))
	 (oh-stride (nth 4 strides))
	 (ow-stride (nth 5 strides))

	 ;; strides on padded-x
	 ;; padded-x = (N C h w)
	 (strides (tensor-stride padded-x))
	 (n-stride-o (nth 0 strides))
	 (c-stride-o (nth 1 strides))
	 (h-stride-o (nth 2 strides))
	 (w-stride-o (nth 3 strides)))
    (declare (type (unsigned-byte 32)
		   n-stride c-stride fh-stride fw-stride oh-stride ow-stride
		   n-stride-o c-stride-o h-stride-o w-stride-o))
    (macrolet ((%* (a b)
		 `(the (unsigned-byte 32) (* (the fixnum ,a) (the fixnum ,b))))
	       (%+ (&rest numbers)
		 `(the (unsigned-byte 32) (+ ,@numbers))))
      (with-facets ((c* (col      :direction 'simple-array :sync t))
		    (x* (padded-x :direction 'simple-array :sync t)))
	(declare (type (simple-array u (*)) c* x*))
	(dotimes (y filter-h) ;; lparallel
	  (let ((y-max (%+ y (%* stride-y out-h))))
	    (dotimes (x filter-w)
	      (let ((x-max (%+ x (%* stride-x out-w))))
		(loop for y-pos fixnum upfrom y below y-max by stride-y for y-pos-abs fixnum upfrom 0 do
		  (loop for x-pos fixnum upfrom x below x-max by stride-x for x-pos-abs fixnum upfrom 0 do
		    (dotimes (n-i N)
		      (dotimes (c-i C)		      
			(setf (aref c* (%+ (%* n-i n-stride)
					   (%* c-i c-stride)
					   (%* y   fh-stride)
					   (%* x   fw-stride)
					   (%* y-pos-abs oh-stride)
					   (%* x-pos-abs ow-stride))) ;; ow-stride=1 when column major
			      (aref x* (%+ (%* n-i n-stride-o)
					   (%* c-i c-stride-o)
					   (%* y-pos h-stride-o)
					   (%* x-pos w-stride-o)))))))))))))
      col)))

(defun call-im2col-kernel (padded-x col N C filter-h filter-w out-h out-w stride-x stride-y)
  "
## [function] call-im2col-kernel

N ... batch
C ... in-channels
filter-h/filter-w kernel-size[0], kernel-size[1]
out-h out-w
stride-x stride-y"
  (funcall (im2col-caller (dtype padded-x))
	   padded-x
	   col
	   N
	   C
	   filter-h
	   filter-w
	   out-h
	   out-w
	   stride-x
	   stride-y))


(defun col2im-kernel (x k-h k-w stride padding)

  )

(define-static-node (Im2ColNode (self N C k-h k-w h-out w-out stride-x stride-y)
		     :slots ((N :initarg :N)
			     (C :initarg :C)
			     (k-h :initarg :k-h)
			     (k-w :initarg :k-w)
			     (h-out :initarg :h-out)
			     (w-out :initarg :w-out)
			     (stride-x :initarg :stride-x)
			     (stride-y :initarg :stride-y))
		     :where (X[N C H W] Col[N C k-h k-w h-out w-out] -> Col[N C k-h k-w h-out w-out])
		     :forward ((self x col)
			       (with-slots ((N N) (C C) (k-h k-h) (k-w k-w) (h-out h-out) (w-out w-out) (stride-x stride-x) (stride-y stride-y)) self
				 (call-im2col-kernel
				  x
				  col
				  n
				  c
				  k-h
				  k-w
				  h-out
				  w-out
				  stride-x
				  stride-y)))
		     :backward ((self dout)
				(values dout nil))))			    


(defun !im2col-cpu (padded-x N C k-h k-w h-out w-out stride-x stride-y)
  "
## [function] !im2col-cpu

N - batch-size
C - in-channels
k-h/k-w kernel-size[0], kernel-size[1] respectively.
h-out w-out
stride-x stride-y - stride[0], stride[1] respectively.
"
  (let* ((col (ax+b `(,N ,C ,k-h ,k-w ,h-out ,w-out) 0 0
		    :order (order padded-x)
		    :dtype (dtype padded-x)))
	 (result (call (Im2ColNode N C k-h k-w h-out w-out stride-x stride-y) padded-x col)))
    (!reshape
     (->contiguous (!permute result 0 4 5 1 2 3))
     (* n h-out w-out)
     t)))


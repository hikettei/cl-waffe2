
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


(define-with-typevar-dense
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
      (proceed
       (!reshape
	(->contiguous (permute* col 0 4 5 1 2 3))
	(%* n (%* out-h out-w))
	t)))))

(defun call-im2col-kernel (padded-x N C filter-h filter-w out-h out-w stride-x stride-y)
  "
## [function] call-im2col-kernel

N ... batch
C ... in-channels
filter-h/filter-w kernel-size[0], kernel-size[1]
out-h out-w
stride-x stride-y"
  (let ((col (ax+b `(,N ,C ,filter-h ,filter-w ,out-h ,out-w) 0 0
		   :dtype (dtype padded-x)
		   :order (order padded-x))))
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
	   stride-y)))


(defun col2im-kernel (x k-h k-w stride padding)

  )


(defun !im2col-cpu ())


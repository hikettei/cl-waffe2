
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


;; TODO: im2col-caller should be moved to ./backends/lisp/
(define-with-typevar
    (im2col-caller u) (padded-x col N C filter-h filter-w out-h out-w stride-x stride-y)
  (declare (optimize (speed 3)) ;; (safety 0)
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

(define-with-typevar
    (∂im2col-caller/∂out-caller u) (dout img-out N C k-h k-w h-out w-out stride-x stride-y)
  (declare (optimize (speed 3)) ;; (safety 0)
           (type AbstractTensor dout img-out)
	   (type (unsigned-byte 32) N C k-h k-w h-out w-out stride-x stride-y))
  ;; dout    ... (N C k-h k-w h-out w-out)
  ;; img-out ... (N C h-out w-out)
  (let* ((strides   (tensor-stride dout))
	 ;; strides on (N C k-h k-w h-out w-out)
	 (n-stride  (nth 0 strides))
	 (c-stride  (nth 1 strides))
	 (kh-stride (nth 2 strides))
	 (kw-stride (nth 3 strides))
	 (oh-stride (nth 4 strides))
	 (ow-stride (nth 5 strides))

	 ;; strides on (N C h-out w-out)
	 (strides (tensor-stride img-out))
	 (n-stride-o (nth 0 strides))
	 (c-stride-o (nth 1 strides))
	 (h-stride-o (nth 2 strides))
	 (w-stride-o (nth 3 strides)))
    (declare (type (unsigned-byte 32)
		   n-stride c-stride kh-stride kw-stride oh-stride ow-stride
		   n-stride-o c-stride-o h-stride-o w-stride-o))
    (macrolet ((%* (a b)
		 `(the (unsigned-byte 32) (* (the fixnum ,a) (the fixnum ,b))))
	       (%+ (&rest numbers)
		 `(the (unsigned-byte 32) (+ ,@numbers))))
      (with-facets ((∂* (dout     :direction 'simple-array :sync t))
		    (i* (img-out  :direction 'simple-array :sync t)))
	(declare (type (simple-array u (*)) ∂* i*))
	(dotimes (y k-h) ;; [TODO] lparallel
	  (let ((y-max (%+ y (%* stride-y h-out))))
	    (dotimes (x k-w)
	      (let ((x-max (%+ x (%* stride-x w-out))))
		(loop for y-pos fixnum upfrom y below y-max by stride-y for y-pos-abs fixnum upfrom 0 do
		  (loop for x-pos fixnum upfrom x below x-max by stride-x for x-pos-abs fixnum upfrom 0 do
		    (dotimes (n-i N)
		      (dotimes (c-i C)
			(dotimes (a h-out)
			  (dotimes (b w-out)
			    ;; img[:, :, y~y_max by stride_y, :, :] += dout[:, :, y, x, :, :]
			    ;; [FixME] setf? incf?
			    (setf
			     (aref i* (%+ (%* n-i n-stride-o)
					  (%* c-i c-stride-o)
					  (%* y-pos h-stride-o)
					  (%* x-pos w-stride-o)))
			     (aref ∂* (%+ (%* n-i n-stride)
					  (%* c-i c-stride)
					  (%* y kh-stride)
					  (%* x kw-stride)
					  (%* a oh-stride)
					  (%* b ow-stride)))))))))))))))
      img-out)))


(defun ∂im2col/∂out (dout img-out N C k-h k-w h-out w-out stride-x stride-y)
  (funcall (∂im2col-caller/∂out-caller (dtype img-out))
	   dout
	   img-out
	   N
	   C
	   k-h
	   k-w
	   h-out
	   w-out
	   stride-x
	   stride-y))

(define-impl (Im2ColNode
	      :device cl-waffe2/backends.lisp:LispTensor
	      :cache-when-compiled nil)
	     :forward ((self x col)
		       (setf (h-of self) (nth 2 (shape x))
			     (w-of self) (nth 3 (shape x)))
		       `(with-slots ((N N) (C C) (k-h k-h) (k-w k-w) (h-out h-out) (w-out w-out) (stride-x stride-x) (stride-y stride-y)) ,self
			  (call-im2col-kernel ,x ,col n c k-h k-w h-out w-out stride-x stride-y))))

(define-impl (Col2ImNode
	      :device cl-waffe2/backends.lisp:LispTensor
	      :cache-when-compiled nil)
	     :forward ((self dout)
		       `(with-slots ((N N) (C C) (k-h k-h) (k-w k-w) (h-out h-out) (w-out w-out) (stride-x stride-x) (stride-y stride-y)) ,self
			  (values (∂im2col/∂out ,dout (img-out-of ,self) N C k-h k-w h-out w-out stride-x stride-y)))))


(defun !im2col (padded-x N C k-h k-w h-out w-out stride-x stride-y)
  "
## [function] !im2col

N - batch-size
C - in-channels
k-h/k-w kernel-size[0], kernel-size[1] respectively.
h-out w-out
stride-x stride-y - stride[0], stride[1] respectively.
"
  (let* ((col (ax+b `(,N ,C ,k-h ,k-w ,h-out ,w-out) 0 0
		    :order (order padded-x)
		    :dtype (dtype padded-x)))
	 (img-out (ax+b `(,N ,C
			     ;; H + 2*pad + stride -1
			     ,(nth 2 (shape padded-x))
			     ,(nth 3 (shape padded-x)))
			0 0
			:order (order padded-x)
			:dtype (dtype padded-x)))
	 (result (call (Im2ColNode N C k-h k-w h-out w-out stride-x stride-y img-out) padded-x col)))
    
    ;;    [N C k-h k-w h-out w-out]
    ;; -> N C k-h k-w h-out w-out
    (call-> result
	    (asnode #'!permute (torch-order 0 4 5 1 2 3))
	    (asnode #'!reshape (* N H-out W-out) t))))


(defun unfold (input dilation kernel-size stride padding)
  "
## [function] unfold

```lisp
(unfold input dilation kernel-size stride padding)
```

Extracts sliding local blocks from a batched input tensor. The detailed specifications follow PyTorch: [nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html).

As of this writing, `input` must be a 4D Tensor even when `N=batch-size=1`.

Corresponding nodes: `cl-waffe2/base-impl:Im2ColNode`, `cl-waffe2/base-impl:Col2ImNode`

### Inputs

Note that `dilation`, `kernel-size`, `stride`, and `padding` are given in this form:

`(list y-direction(Height) x-direction(Width))`

`input[AbstractTensor]` the tensor to be unfold.

`dilation[list]` a parameter that controls the stride of elements within the neighborhood.

`kernel-size[list]` the size of sliding blocks.

`padding[list]` implicts the number of zero-padding to be added on both sides of input.

`stride[list]` the number of stride of the sliding blocks.
"

  (multiple-value-bind (N C H-in W-in) (apply #'values (shape input))
    (let* ((H-out (cl-waffe2/nn::conv-out-size H-in (car padding)    (car dilation) (car kernel-size) (car stride)))
	   (W-out (cl-waffe2/nn::conv-out-size W-in (second padding) (car dilation) (second kernel-size) (second stride)))
	   (p-y (mod H-out (second stride)))
	   (p-x (mod W-out (car stride))))

      (call-> input
	      (asnode #'padding    `(t t (,(second padding) ,(+ (second padding) p-y)) (,(car padding) ,(+ (car padding) p-x))))
	      (asnode #'!im2col N C (second kernel-size) (car kernel-size) h-out w-out (car stride) (second stride))))))




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

(defun !im2col (padded-x N C k-h k-w h-out w-out stride-h stride-w padding-h padding-w dilation-h dilation-w)
  "
## [function] !im2col

N - batch-size
C - in-channels
k-h/k-w kernel-size[0], kernel-size[1] respectively.
h-out w-out
stride-x stride-y - stride[0], stride[1] respectively.
"
  (let* ((col (make-input `(,N ,C ,k-h ,k-w ,h-out ,w-out) nil
			  :order (order padded-x)
			  :dtype (dtype padded-x)))
	 ;; img-out is a tensor for future backward computation
	 (img-out (make-input `(,N ,C
				   ;; H + 2*pad + stride -1
				   ,(nth 2 (shape padded-x))
				   ,(nth 3 (shape padded-x)))
			      nil
			      :order (order padded-x)
			      :dtype (dtype padded-x)))
	 (result (call (Im2ColNode
			N C k-h k-w
			h-out w-out
			stride-h stride-w
			padding-h padding-w
			dilation-h dilation-w
			img-out)
		       padded-x col)))
    
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
	   (pad-h (mod H-out (car    stride)))
	   (pad-w (mod W-out (second stride))))

      (call-> input
	      (asnode #'padding `(t t (,(car padding) ,(+ (car padding) pad-h)) (,(second padding) ,(+ (second padding) pad-w))))
	      (asnode #'!im2col
		      N C (car kernel-size) (second kernel-size)
		      h-out w-out (car stride) (second stride)
		      (car padding) (second padding)
		      (car dilation) (second dilation))))))




(in-package :cl-waffe2/nn)

(defmodel (BatchNorm (self in-features &key (affine t) (eps 1e-5))
	   :documentation "Applies Batch Normalization over a 4D input (N C H W) as described in the paper [Batch Normalization](https://arxiv.org/abs/1502.03167).

```math
BatchNorm(x) = \\frac{x - E[x]}{\\sqrt{Var[x] + ε}}\\times{γ}+β
```

### Inputs

`in-features[fixnum]` - C from an excepted input size (N C H W)

`affine[bool]` Set T to apply affine transofmration to the output. In default, set to t.

`eps[single-float]` a value added to the denominator for numerical stability. Default: 1e-5.

### Parameters

`alpha` (in-features) is a trainable tensor filled with `1.0`. accessor: `alpha-of`

`beta`  (in-features) is a trainable tensor filled with `0.0`. accessor: `beta-of`
"
	   :slots ((alpha :initform nil :accessor alpha-of)
		   (beta  :initform nil :accessor beta-of)
		   (eps   :initform nil :initarg :eps :accessor eps-of))
	   :where (X[N in-features H W] -> OUT[N in-features H W])
	   :on-call-> ((self x)
		       (with-slots ((alpha alpha) (beta beta)) self
			 (let* ((m (->contiguous (!mean x :axis 1 :keepdims t)))
				(l (!- x m))
				(d (->contiguous (!mean (!expt l 2) :axis 1 :keepdims t)))
				(r (!div l (!sqrt (!+ d (eps-of self))))))
			   (if (and alpha beta)
			       (call-> r
				       (asnode #'!mul (%transform alpha[i] -> [~ i]))
				       (asnode #'!add (%transform beta[i]  -> [~ i])))
			       r)))))
  (when affine
    (setf (alpha-of self) (parameter (ax+b `(,in-features) 0 1))
	  (beta-of  self) (parameter (ax+b `(,in-features) 0 0)))))

(defmodel (LayerNorm (self normalized-shape &key (eps 1.0e-5) (affine T))
	   :slots ((alpha :initform nil :accessor alpha-of)
		   (beta  :initform nil :accessor beta-of)
		   (shape :initform nil :initarg :normalized-shape :accessor dim-of)
		   (eps   :initform nil :initarg :eps :accessor eps-of))
	   :documentation "Applies Layer Normalization over a mini-batch of inputs as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450)

```math
LayerNorm(x) = \\frac{x - E[x]}{\\sqrt{Var[x] + ε}}\\times{γ}+β
```

The mean and standard-deviation are calculated over the last D dimension where D = `(length normalized-shape)`. The parameters β and γ are trainable affine transforms created if `affine` is set to T.

### Inputs

`normalized-shape` [list or fixnum] the size of kernel

`eps[single-float]` a value added to the denominator for the numerical stability.

`affine[boolean]` Set T to use affine transformation.

### Parameters

`alpha` (normalized-shape) is a trainable tensor filled with `1.0`. accessor: `alpha-of`

`beta` (normalized-shape) is a trainable tensor filled with `0.0`. accessor: `beta-of`
"
	   :where (X[~ normalized-shape] -> out[~ normalized-shape])
	   :on-call->  ((self X)
			(with-slots ((alpha alpha) (beta beta)) self
			  (let* ((last-dim (length (dim-of self)))
				 (u (!mean x :axis (- last-dim) :keepdims t))
				 (s (!mean (!expt (!sub x u) 2) :axis (- last-dim) :keepdims t))
				 (x (!div (!sub x u)
					  (!sqrt (!add (->contiguous s) (eps-of self))))))

			    (if (and alpha beta)
				(!add (!mul x (!flexible alpha)) (!flexible beta))
				x)))))
  (when affine
    (setf (alpha-of self) (parameter (ax+b `(,@normalized-shape) 0 1))
	  (beta-of  self) (parameter (ax+b `(,@normalized-shape) 0 0)))))



(in-package :cl-waffe2/nn)

;; kiokutigaikamo atode zissou siraberu
(defmodel (BatchNorm (self in-features &key (affine t) (eps 1e-5))
	   :documentation ""
	   :slots ((alpha :initform nil :accessor alpha-of)
		   (beta  :initform nil :accessor beta-of)
		   (shape :initform nil :initarg :in-features :accessor shape-of)
		   (eps   :initform nil :initarg :eps :accessor eps-of))
	   :on-call-> ((self x)
		       (with-slots ((alpha alpha) (beta beta)) self
			 (let* ((dim (1- (length (shape-of self))))
				(m (!mean x :axis dim))
				(mp (!sub x m))
				(d (!mean (!expt mp 2) :axis dim))
				(r (!div mp (!sqrt (!add d (eps-of self))))))
			   (if (and alpha beta)
			       (!add (!mul r (!flexible alpha)) (!flexible beta))
			       r)))))
  
   (when affine
     (setf (alpha-of self) (parameter (ax+b `(,@in-features) 0 1))
	   (beta-of  self) (parameter (ax+b `(,@in-features) 0 0)))))
			       
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


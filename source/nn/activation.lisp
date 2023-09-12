
;; Non Linear Activations

(in-package :cl-waffe2/nn)

(declaim (ftype (function (AbstractTensor) AbstractTensor)
		!relu
		!sigmoid
		!gelu))

(defun !relu (x)
  "
## [function] !relu

```lisp
(!relu x)
```

Computes ReLU to the given tensor.

```math
ReLU(x) = max(x, 0)
```
"
  (declare (type AbstractTensor x))
  (!mul x (A>scal x 0.0)))

(defun !sigmoid (x)
  "
## [function] !sigmoid

```lisp
(!sigmoid x)
```

Computes sigmoid function to the given tensor.

```math
Sigmoid(x) = \\frac{1}{1 + exp(-x)}
```
"
  (declare (type AbstractTensor x))
  (!div 1 (!add 1 (!exp (!mul -1 x)))))

(defun !gelu (x)
  "
## [function] !gelu

```lisp
(!gelu x)
```

Applies the Gaussian Error Linear Units function approximated with:

```math
GeLU(x) = 0.5\\times{x}\\times{(1 + Tanh(\\sqrt{\\frac{2}{π}}\\times{(x + 0.44715\\times{x^3})}))}
```

"
  (declare (type AbstractTensor x))
  (!* 0.5 x
      (!+ 1
	  (!tanh
	   (!* (coerce (sqrt (/ 2.0 pi)) (dtype->lisp-type (dtype x)))
	       (!+ x
		   (!* 0.044715 (!expt x 3))))))))

(declaim (ftype (function (AbstractTensor &key (:negative-slope single-float)) AbstractTensor) !leakey-relu))
(defun !leakey-relu (x &key (negative-slope 0.01))
  "
## [function] !leakey-relu

```lisp
(!leakey-relu x &key (negative-slope 0.01))
```

Applies the element-wise function:

```lisp
LeakeyReLU(x) = max(x, 0) + negative-slope\\times{min(0, x)}
```

### Inputs

`x[AbstractTensor]`

`negative-slope[single-float]`
"
  (declare (type AbstractTensor x)
	   (type single-float negative-slope))

  (let ((mask (A>=scal x 0.0
		       :true-then 1.0
		       :false-then negative-slope)))
    (!mul x mask)))

(declaim (ftype (function (AbstractTensor &key (:alpha single-float)) AbstractTensor) !elu))
(defun !elu (x &key (alpha 1.0))
  "
## [function] !elu

```lisp
(!elu x &key (alpha 1.0))
```

Applies the Expotential Linear Units Function (ELUs) element-wise as described in [this paper](https://arxiv.org/abs/1511.07289)

```math
\\begin{equation}
  ELU(x)=
  \\begin{cases}
    \\text{x} & if x>0 \\\\
    \\text{α*(exp(x)-1)} & \\text{otherwise}
  \\end{cases}
\\end{equation}
```
"
  ;; [TODO] Fusion
  (let* ((mask1 (A>scal x 0 :true-then 0 :false-then 1))
	 (mask2 (A>scal x 0 :true-then 1 :false-then 0))
	 (out1  (!* mask1 alpha (!- (!exp x) 1)))
	 (out2  (!* mask2 x)))
    (!add out1 out2)))


(defun !softmax (x &key (avoid-overflow t) (axis 1))
  "
## [function] !softmax

```lisp
(!softmax x &key (avoid-overflow t) (axis 1))
```

Returns a tensor that applied Softmax function along the given axis.

```lisp
Softmax(x_i) = exp(x_i)\\div{sum(x_j, axis)}
```

If avoid-overflow is set to t:

```lisp
x_i = x_i - mean(x)
```

### Inputs

`avoid-overflow[boolean]` If t, `exp(x_i)` is substracted by the mean value of `x`.

`axis[fixnum or list or t]` The axis to be reducted.
"

  (if avoid-overflow
      (let* ((x1    (!sub x (!mean x  :axis axis :keepdims t)))
	     (expx1 (!exp x1))
	     (z     (!sum   expx1 :axis axis :keepdims t)))
	(!div expx1 z))
      (let ((x1 (!exp x)))
	(!div x1 (!sum x1 :axis axis :keepdims t)))))



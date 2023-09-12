
;; Non Linear Functions

(in-package :cl-waffe2/nn)

;; [TODO] ドキュメント更新 + Examples + Package追加 + Test !mulとかの型宣言

;; Softmax
;; ReLU
;; GeLU
;; Leakey-ReLU
;; Swish Hardswish Hardtanh

(declaim (ftype (function (AbstractTensor) AbstractTensor)
		!relu
		!sigmoid
		!gelu))

(declaim (ftype (function (AbstractTensor &key (:negative-slope single-float)) AbstractTensor)
		!leakey-relu))

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




(defun !softmax (x &key (avoid-overflow t) (axis 1))
  "
## [function] !softmax

Returns a tensor that applied Softmax function.

```lisp
Softmax(x_i) = exp(x_i)\\div{sum(x_j, axis)}
```

### Inputs

`avoid-overflow[boolean]` If t, `exp(x_i)` is substracted by the mean value of `x`.

`axis[fixnum or list or t]` The axis to be reducted.
"

  (if avoid-overflow
      (let* ((x1 (!sub x (!mean x  :axis axis :keepdims t)))
	     (z  (!sum   (!exp x1) :axis axis :keepdims t)))
	(!div (!exp x1) z))
      (!div (!exp x) (!sum (!exp x) :axis axis :keepdims t))))



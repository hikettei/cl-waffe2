
(in-package :cl-waffe2/nn)

;; Softmax
;; ReLU
;; GeLU
;; Leakey-ReLU
;;

(defun !relu (x)
  "
## [function] !relu

Returns a tensor that applied ReLU element-wise.

```math
ReLU(x) = max(x, 0)
```"

  (!mul x (A>scal x 0.0)))

(defun !sigmoid (x)
  "
## [function] !sigmoid

Returns a tensor that applied sigmoid function element-wise.

```math
Sigmoid(x) = \\frac{1}{1 + exp(-x)}
```
"
  
  (!div 1 (!add 1 (!exp (!mul -1 x)))))

(defun !gelu (x)
  "
## [function] !gelu

Approximates GeLU (Not tested yet): `(!* 0.5 x (!+ 1 (!tanh (!* (sqrt (/ 2 pi)) (!+ x (!* 0.044715 (!expt x 3)))))))`
"
  ;; I dunno if this is really works
  (!* 0.5 x (!+ 1 (!tanh (!* (sqrt (/ 2 pi)) (!+ x (!* 0.044715 (!expt x 3))))))))

;; todo (!matmul !t !t) test
;; Bug: (Proceed (!sum (Proceed (!Softmax x))))
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


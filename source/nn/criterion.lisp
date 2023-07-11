
(in-package :cl-waffe2/nn)

;; TO DO:
;; L1
;; BinaryCrossEntropy
;; KLdiv
;; CosineSim

;; MSE
;; SoftMaxCrossEntropy
;; BSE

(defun L1Norm (x y &key (reduction :mean))
  "
## [function] L1Norm

```
(L1Norm x p &key (:reduction :mean))
```

Returns a tensor that measures L1 Norm between each element in the input `x` and `y`.

```math
l(x, y) = L = {l_1, ..., l_n}^\\intercal, l_n = abs(x_n - y_n)
```

where `N` is a batch-size.

In addition, reading the value of a `:reduction` keyword (one of `:mean` `:sum` `nil`), the result of `L` is reducted. (If nil, reduction is ignored.)
"
  (declare (type AbstractTensor x y)
	   (type (or nil (member :mean :sum))))
  
  (let ((l (!sub x y)))
    (case reduction
      (:sum
       (!sum (!abs l)))
      (:mean
       (!mean (!abs l)))
      (T (!abs l)))))

(defun mse (x y &key (reduction :mean))
  "
## [function] mse
```
(mse x p &key (:reduction :mean))
```
Returns a tensor that measures the MSE error (L2Norm) between each element in the input `x` and `y`.

```math
l(x, y) = L = {l_1, ..., l_n}^\\intercal, l_n = (x_n - y_n)^2
```

where `N` is a batch-size.

In addition, reading the value of a `:reduction` keyword (one of `:mean` `:sum` `nil`), the result of `L` is reducted. (If nil, reduction is ignored.)
"
  (declare (type AbstractTensor x y)
	   (type (or nil (member :mean :sum))))
  
  (let ((l (!sub x y)))
    (case reduction
      (:sum
       (!sum (!mul l l)))
      (:mean
       (!mean (!mul l l)))
      (T (!mul l l)))))

(defun cross-entropy-loss (x labels &key (eps 1e-7))
  "
## [fucntion] cross-entropy-loss

```lisp
(cross-entropy-loss x labels &key (eps 1e-y))
```
Returns a tensor that measures the Cross-Entropy-Error between each element in the x and labels
"
  )


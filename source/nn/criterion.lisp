
(in-package :cl-waffe2/nn)

;; TODO: ================================
;; L1/one-hot
;; BinaryCrossEntropy
;; KLdiv
;; CosineSim (rather than distance.lisp?)

;; MSE
;; SoftMaxCrossEntropy
;; BSE
;; ======================================

(deftype reduction-opt-t ()
  "Keywords indicating the option to reduct"
  `(member :mean :sum t))

(defgeneric L1Norm (x y &key reduction))
(defmethod L1Norm (x y &key (reduction t))
  "
## [function] L1Norm

```
(L1Norm x p &key (:reduction t))
```

Returns a tensor that measures L1 Norm between each element in the input `x` and `y`.

```math
l(x, y) = L = {l_1, ..., l_n}^\\intercal, l_n = abs(x_n - y_n)
```

where `N` is a batch-size.

In addition, reading the value of a `:reduction` keyword (one of `:mean` `:sum` `t`), the result of `L` is reducted. (If t, reduction is ignored.)
"
  (declare (type AbstractTensor x y)
	   (type reduction-opt-t reduction))
  
  (let ((l (!sub x y)))
    (case reduction
      (:sum
       (!sum  (!abs l)))
      (:mean
       (!mean (!abs l)))
      (T (!abs l)))))

(defgeneric mse (x y &key reduction))
(defmethod mse (x y &key (reduction t))
  "
## [function] mse
```
(mse x p &key (:reduction T))
```

Returns a tensor that measures the MSE error (i.e.: L2Norm) between each element in the input `x` and `y`.

```math
l(x, y) = L = {l_1, ..., l_n}^\\intercal, l_n = (x_n - y_n)^2
```

where `N` is a batch-size.

In addition, reading the value of a `:reduction` keyword (one of `:mean` `:sum` `t`), the result of `L` is reducted. (If t, this operation is ignored.)
"
  (declare (type AbstractTensor x y)
	   (type reduction-opt-t reduction))
  
  (let ((l (!sub x y)))
    (case reduction
      (:sum
       (!sum  (!mul l l)))
      (:mean
       (!mean (!mul l l)))
      (T      (!mul l l)))))

(node->defun sce-diff-static
    (Dy[~ length n-dimension] z[~ length n-dimension] Labels[~ length n-dimension] Batch-Size[scal]
     ->
     X.grad[~ length n-dimension] where scal = 1)
  (let* ((z  (!sub z labels))
	 (dx (!div (!mul dy z) batch-size)))
    dx))

(define-op (Softmax-Cross-Entropy-Node (self &key (axis 1) (delta 1e-7) (avoid-overflow t))
	    :slots ((softmax       :initform nil :accessor softmax-of)
		    (cross-entropy :initform nil :accessor celoss-of))
	    
	    :save-for-backward-names (z label)
	    :where (X[~ length n-dimension] Labels[~ length n-dimension] -> OUT[~ length n-dimension])
	    :forward ((self x label)
		      (let ((z (funcall (softmax-of self) x)))
			(with-setting-save4bw ((z z) (label label)) self
			  (funcall (celoss-of self) z label))))
	    :backward ((self dout)
		       (with-reading-save4bw ((z z) (label label)) self
			 (sce-diff-static
			  dout
			  z
			  label
			  (make-tensor (car (last (shape z) 2)) :dtype (dtype x) :order (order x))))))
  
  ;; Initializing/Compiling static functions in advance:
  (let ((softmax (node->lambda (A[~] -> B[~])
		   (!softmax a :axis axis :avoid-overflow avoid-overflow)))
	(celoss  (node->lambda (A[~] B[~] -> OUT[~])
		   (cross-entropy-loss a b :delta delta))))

    (setf (softmax-of self) softmax
	  (celoss-of  self) celoss)))
	    

(defgeneric cross-entropy-loss (x labels &key delta reduction))
(defmethod cross-entropy-loss (x labels &key (delta 1e-7) (reduction t))
  "
## [fucntion] cross-entropy-loss

```lisp
(cross-entropy-loss x labels &key (delta 1e-7) (reduction t))
```

Returns a tensor that measures the Cross-Entropy-Error between each element in the x and labels.

```math
L_i = -p_ilog(x_i + delta)
```

```math
\\begin{equation}
  out_i=
  \\begin{cases}
    sum(L)  & \\text{reduction = sum} \\\\
    mean(L) & \\text{reduction = mean} \\\\
    L       & \\text{otherwise}
  \\end{cases}
\\end{equation}
```

### Inputs

`x[AbstractTensor]`

`labels[AbstractTensor]` one-hot encoding.

`reduction` one of :sum :mean t
"
  (declare (type AbstractTensor x labels)
	   (type reduction-opt-t reduction))

  ;; KLDiv: xlogp
  (let ((z (!mul -1 (!mul labels (!loge (!add x delta))))))
    (case reduction
      (:sum  (!sum z))
      (:mean (!mean z))
      (T     z))))

(defgeneric softmax-cross-entropy (x labels &key axis delta avoid-overflow reduction))
(defmethod softmax-cross-entropy (x labels &key (axis 1) (delta 1e-7) (avoid-overflow nil) (reduction t))
  "
## [function] softmax-cross-entropy

```lisp
(softmax-cross-entropy x labels &key (axis 1) (delta 1e-7) (avoid-overflow nil) (reduction t))
```

Returns a tensor that measures the Softmax-Cross-Entropy-Error between each element in the x and labels.

```math
out = CrossEntropyLoss(Softmax(x), labels)
```

### Inputs

`x[AbstractTensor]` distribution to measure

`labels[AbstractTensor]` answer labels with one-hot encoding.
"
  (declare (type reduction-opt-t reduction))
  (let ((out (call (Softmax-Cross-Entropy-Node :axis axis :delta delta :avoid-overflow avoid-overflow) x labels)))
    (case reduction
      (:sum (!sum out))
      (:mean (!mean out))
      (T out))))

(defgeneric ->one-hot (x))
(defmethod ->one-hot (x)
  "
## [function] ->one-hot

Creates an one-hot encoding tensor from the given x
"

  )

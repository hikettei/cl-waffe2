
(in-package :cl-waffe2/base-impl)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

;; Memo: uint8 with sin, construct lut?

;; ==============================================================
;; F() -> G(X, OUT) Function Family
;; ==============================================================
(macrolet ((define-elwise-node ((name lisp-func) &optional save-for-backward backward)
	     (let ((node-name  (symb name 'node))
		   (scal-node-name (symb 'Scalar- name 'node))
		   (defun-name (symb '! name)))
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',node-name)
		(export ',scal-node-name)
		(export ',defun-name)
		(defnode (,node-name (myself)
			  :where (X[~] OUT[~] -> OUT[~])
			  :save-for-backward ,save-for-backward
			  :backward ,backward
			  :documentation
			  ,(format nil "The node `~a` takes X as an argument, applying a ~(~a~) function into each element and writes the result into out.

```math
OUT\\gets{~(~a~)(X)}
```

save-for-backward: ~a

See also: `~a` `~(~a~)`"
				   node-name
				   name
				   name
				   save-for-backward
				   scal-node-name
				   defun-name)))
		(defnode (,scal-node-name (myself)
			  :where (X[~] OUT[~] -> OUT[~])
			  :save-for-backward ,save-for-backward
			  :backward ,backward
			  :out-scalar-p t
			  :documentation
			  ,(format nil "The node ~a takes scalar X as an argument, applying a ~(~a~) function into each element and writes the result into out.

```math
out\\gets{~(~a~)(x)}
```
save-for-backward: ~a

See also: `~a` `~(~a~)`"
				   scal-node-name
				   name
				   name
				   save-for-backward
				   node-name
				   defun-name)))

		(define-impl (,scal-node-name :device ScalarTensor)
			     :save-for-backward ,save-for-backward
			     :forward ((self x out)
				       (let ((caller ,lisp-func))
					 `(progn
					    (setf (tensor-vec ,out) (funcall ,caller (tensor-vec ,x)))
					    ,out))))
			     
		(defun ,defun-name (x &key (-> nil))
		  ,(format nil "
## [function] ~(~a~)

```lisp
(~(~a~) x &key (-> nil))
```

The function ~(~a~) takes `x` as an argument, applying a ~(~a~) function into each element and writes the result into `->`.

```math
OUT_{copy}\\gets{~(~a~)(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`~a` `~a`

### SideEffects

`->` is destructed.
"
			   defun-name
			   defun-name
			   defun-name
			   name
			   name
			   scal-node-name
			   node-name)
		  (if (or (numberp x) (scalar-p x))
		      (let ((x (if (numberp x)
				   (make-tensor x)
				   x)))
			(if ->
			    (forward (,scal-node-name) x ->)
			    (forward (,scal-node-name) x (!copy x))))
		      (if ->
			  (forward (,node-name) x ->)
			  (forward (,node-name) x (!copy x :maybe-in-place t)))))))))

  ;; define-elwise-node will define: nameNode, !name.
  ;; To Investigate: the function !abs consumes 2x time larger memory than required.
  
  (define-elwise-node (abs #'abs)
      (t nil)
    ((self dout dx dy)
     (declare (ignore dy))
     (values (!mul dout (!sign dx)) nil)))
  
  (define-elwise-node (sign #'signum)
      (t nil)
    ((self dout dx dy)
     (declare (ignore dout dy))
     ;; (sign)' = 0
     (values (!mul dx 0) nil)))
  
  (define-elwise-node (sqrt #'sqrt)
      (t nil)
    ((self dout dx dy)
     (declare (ignore dy))
     ;; ∂dout/∂dx sqrt(x) = 1 / sqrt(x)
     ;; ∂dout/∂dy sqrt(x) = 0
     (values (!mul dout (!div 1 dx)) nil)))
  
  (define-elwise-node (square #'(lambda (x) (* x x)))
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x square(x) => x
     (values (!mul dout x) nil)))
  
  (define-elwise-node (sin #'sin)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values (!mul dout (!cos x)) nil)))
  
  (define-elwise-node (cos #'cos)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values (!mul dout (!mul -1 (!sin x))) nil)))
  
  (define-elwise-node (tan #'tan)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x tan(x) = 1/(cos(x)^2)
     (values (!mul dout (!div 1 (!mul (!cos x) (!cos x)))) nil)))

  
  (define-elwise-node (sinh #'sinh)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values (!mul dout (!cosh x)) nil)))
  
  (define-elwise-node (cosh #'cosh)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values (!mul dout (!mul -1 (!sinh x))) nil)))
  
  (define-elwise-node (tanh #'tanh)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x tan(x) = 1/(cos(x)^2)
     (values (!mul dout (!div 1 (!mul (!cosh x) (!cosh x)))) nil)))


  (define-elwise-node (asin #'asin)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x asin(x) = 1/sqrt(1-x^2)
     (values (!mul dout (!div 1 (!sqrt (!sub 1 (!square x))))) nil)))
  
  (define-elwise-node (acos #'acos)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x acos(x) = -1/sqrt(1-x^2)
     (values (!mul dout (!div -1 (!sqrt (!sub 1 (!square x))))) nil)))
  
  (define-elwise-node (atan #'atan)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x atan(x) = 1/(1+x^2)
     (values (!mul dout (!div 1 (!add 1 (!square x)))) nil)))

  ;; keisan mendoi...
  
  (define-elwise-node (asinh #'asinh))
  (define-elwise-node (acosh #'acosh))
  (define-elwise-node (atanh #'atanh))

  (define-elwise-node (exp #'exp)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values (!mul dout (!exp x)) nil)))
  
  (define-elwise-node (log2 #'(lambda (x) (log x 2)))
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values
      (!mul dout (!div 1 (!mul x (log 2))))
      nil)))
  
  (define-elwise-node (log10 #'(lambda (x) (log x 10)))
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values
      (!mul dout (!div 1 (!mul x (log 10))))
      nil)))
  
  (define-elwise-node (logE #'log)
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values
      (!mul dout (!div 1 x))
      nil))))

;; ==============================================================
;; F() -> G(X, N, OUT) Function Family
;; ==============================================================

(eval-when (:compile-toplevel :load-toplevel :execute)

;; TODO Expt for Scalar.
(export '(ExptNode !expt))
(defnode (ExptNode (myself)
	  :where (X[~] OUT[~] N[scal] -> OUT[~] where scal = 1)
	  :backward ((self dout x out n)
		     (declare (ignore out))
		     (values
		      (!mul dout (!mul n (!expt x (!sub n 1))))
		      nil
		      (!mul dout (!mul (!expt x n) (!loge x)))))
	  :documentation "The node ExptNode applies (expt X N) into each element, writing the result into out."))

(defun !expt (x n &key (-> nil))
  "The function !expt applies (expt X N) into each element, writing the result into out.

Inputs:
    N - ScalarTensor
    X - AbstractTensor
    Out - AbstractTensor or nil

Output:
   out - AbstractTensor"
  (declare (type AbstractTensor x))
  (let ((n (if (numberp n)
	       (make-tensor n)
	       n)))
    (if ->
	(forward (ExptNode) x -> n)
	(forward (ExptNode) x (!copy x :maybe-in-place t) n))))

;; !pow

)



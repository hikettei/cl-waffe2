
(in-package :cl-waffe2/base-impl)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

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
			  :backward ,backward
			  :documentation
			  ,(format nil "The node ~(~a~) takes X as an argument, applying a ~(~a~) function into each element and writes the result into out.
save-for-backward: ~a"
				   node-name
				   name
				   save-for-backward)))
		(defnode (,scal-node-name (myself)
			  :where (X[~] OUT[~] -> OUT[~])
			  :backward ,backward
			  :out-scalar-p t
			  :documentation
			  ,(format nil "The node ~(~a~) takes scalar X as an argument, applying a ~(~a~) function into each element and writes the result into out.
save-for-backward: ~a"
				   scal-node-name
				   name
				   save-for-backward)))

		(define-impl (,scal-node-name :device ScalarTensor)
			     :save-for-backward ,save-for-backward
			     :forward ((self x out)
				       (let ((caller ,lisp-func))
					 `(progn
					    (setf (tensor-vec ,out) (funcall ,caller (tensor-vec ,x)))
					    ,out))))
			     
		(defun ,defun-name (x &key (-> nil))
		  ,(format nil "The function ~(~a~) takes X as an argument, applying a ~(~a~) function into each element and writes the result into out.

Inputs:
- x   (AbstractTensor)
- -> (nil or AbstractTensor). If nil, a new tensor is allocated.

Return:
- -> (AbstractTensor)

SideEffects:
- -> will be destructed."
			   defun-name
			   name)
		  (if (or (numberp x) (scalar-p x))
		      (let ((x (if (numberp x)
				   (make-tensor x)
				   x)))
			(if ->
			    (forward (,scal-node-name) x ->)
			    (forward (,scal-node-name) x (!copy x))))
		      (if ->
			  (forward (,node-name) x ->)
			  (forward (,node-name) x (!copy x)))))))))

  ;; define-elwise-node will define: nameNode, !name.

  ;; To Investigate, the function !abs consumes 2x time larger memory than required.
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
     (values (!mul dout (!div 1 (!expt 2 (!cos x)))) nil)))

  
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
     (values (!mul dout (!div 1 (!expt 2 (!cosh x)))) nil)))


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
		      (!mul dout (!mul (!expt x n)
				       (!loge x)))))
	  :documentation "The node ExptNode applies (expt N X) into each element, writing the result into out."))

(defun !expt (n x &key (-> nil))
  "The function !expt applies (expt N X) into each element, writing the result into out.

Inputs:
    N - ScalarTensor
    X - AbstractTensor
    Out - AbstractTensor or nil

Output:
   out - AbstractTensor"
  (let ((n (if (numberp n)
	       (make-tensor n)
	       n)))
    (if ->
	(forward (ExptNode) x -> n)
	(forward (ExptNode) x (make-input (shape x) nil :dtype (dtype x) :order (order x)) n))))

)



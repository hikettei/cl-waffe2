
(in-package :cl-waffe2/base-impl)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

;; ==============================================================
;; F() -> G(X, OUT) Function Family
;; ==============================================================
(macrolet ((define-elwise-node (name &optional save-for-backward backward)
	     (let ((node-name  (symb name 'node))
		   (defun-name (symb '! name)))
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',node-name)
		(export ',defun-name)
		(defnode (,node-name (myself)
			  :where `(X[~] OUT[~] -> OUT[~])
			  :backward ,backward
			  :documentation
			  ,(format nil "The node ~(~a~) takes X as an argument, applying a ~(~a~) function into each element and writes the result into out.
save-for-backward: ~a"
				   node-name
				   name
				   save-for-backward)))
		(defun ,defun-name (x &key (out nil))
		  ,(format nil "The function ~(~a~) takes X as an argument, applying a ~(~a~) function into each element and writes the result into out.

Inputs:
- x   (AbstractTensor)
- out (nil or AbstractTensor). If nil, a new tensor is allocated.

Return:
- out (AbstractTensor)

SideEffects:
- out will be destructed."
			   defun-name
			   name)
		  (if out
		      (forward (,node-name) x out)
		      (forward (,node-name) x (!copy x))))))))

  ;; define-elwise-node will define: nameNode, !name.

  ;; To Investigate, the function !abs consumes 2x time larger memory than required.
  (define-elwise-node abs
      (t nil)
    ((self dout dx dy)
     (declare (ignore dy))
     (values (!mul dout (!sign dx)) nil)))
  
  (define-elwise-node sign
      (t nil)
    ((self dout dx dy)
     (declare (ignore dout dy))
     ;; (sign)' = 0
     (values (!mul dx 0) nil)))
  
  (define-elwise-node sqrt
      (t nil)
    ((self dout dx dy)
     (declare (ignore dy))
     ;; ∂dout/∂dx sqrt(x) = 1 / sqrt(x)
     ;; ∂dout/∂dy sqrt(x) = 0
     (values (!mul dout (!div 1 dx)) nil)))
  
  (define-elwise-node square
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x square(x) => x
     (values (!mul dout x) nil)))
  
  (define-elwise-node sin
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values (!mul dout (!cos x)) nil)))
  
  (define-elwise-node cos
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values (!mul dout (!mul -1 (!sin x))) nil)))
  
  (define-elwise-node tan
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x tan(x) = 1/(cos(x)^2)
     (values (!mul dout (!div 1 (!expt 2 (!cos x)))) nil)))

  
  (define-elwise-node sinh
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values (!mul dout (!cosh x)) nil)))
  
  (define-elwise-node cosh
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values (!mul dout (!mul -1 (!sinh x))) nil)))
  
  (define-elwise-node tanh
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x tan(x) = 1/(cos(x)^2)
     (values (!mul dout (!div 1 (!expt 2 (!cosh x)))) nil)))


  (define-elwise-node asin
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x asin(x) = 1/sqrt(1-x^2)
     (values (!mul dout (!div 1 (!sqrt (!sub 1 (!square x))))) nil)))
  
  (define-elwise-node acos
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x acos(x) = -1/sqrt(1-x^2)
     (values (!mul dout (!div -1 (!sqrt (!sub 1 (!square x))))) nil)))
  
  (define-elwise-node atan
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     ;; ∂out/∂x atan(x) = 1/(1+x^2)
     (values (!mul dout (!div 1 (!add 1 (!square x)))) nil)))

  ;; keisan mendoi...
  
  (define-elwise-node asinh)
  (define-elwise-node acosh)
  (define-elwise-node atanh)

  (define-elwise-node exp
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values (!mul dout (!exp x)) nil)))
  
  (define-elwise-node log2
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values
      (!mul dout (!div 1 (!mul x (log 2))))
      nil)))
  
  (define-elwise-node log10
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values
      (!mul dout (!div 1 (!mul x (log 10))))
      nil)))
  
  (define-elwise-node logE
      (t nil)
    ((self dout x out)
     (declare (ignore out))
     (values
      (!mul dout (!div 1 x))
      nil))))

;; ==============================================================
;; F() -> G(X, N, OUT) Function Family
;; ==============================================================

(defnode (ExptNode (myself)
	  :where `(X[~] OUT[~] N[scal] -> OUT[~] where scal = 1)
	  :backward ((self dout x out n)
		     (declare (ignore out))
		     (values
		      (!mul dout (!mul n (!expt x (!sub n 1))))
		      nil
		      (!mul dout (!mul (!expt x n)
				       (!loge x)))))
	  :documentation "The node ExptNode applies (expt N X) into each element, writing the result into out."))

(defun !expt (n x &key (out nil))
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
    (if out
	(forward (ExptNode) x out n)
	(forward (ExptNode) x (make-input (shape x) nil :dtype (dtype x) :order (order x)) n))))



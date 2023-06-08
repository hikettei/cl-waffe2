
(in-package :cl-waffe2/base-impl)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

;; ==============================================================
;; F() -> G(X, OUT) Function Family
;; ==============================================================
(macrolet ((define-elwise-node (name)
	     (let ((node-name  (symb name 'node))
		   (defun-name (symb '! name)))
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',node-name)
		(export ',defun-name)
		(defnode (,node-name (myself)
			  :where `(X[~] OUT[~] -> OUT[~])
			  :documentation
			  ,(format nil "The node ~(~a~) takes X as an argument, applying a ~(~a~) function into each element and writes the result into out."
				   node-name
				   name)))
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
		      (forward (,node-name)
			       x
			       (make-input (shape x) nil
					   :dtype (dtype x)
					   :order (order x)))))))))

  ;; define-elwise-node will define: nameNode, !name.
  (define-elwise-node abs)
  (define-elwise-node sqrt)
  (define-elwise-node square)
  
  (define-elwise-node sin)
  (define-elwise-node cos)
  (define-elwise-node tan)

  (define-elwise-node asin)
  (define-elwise-node acos)
  (define-elwise-node atan)

  (define-elwise-node sinh)
  (define-elwise-node cosh)
  (define-elwise-node tanh)

  (define-elwise-node asinh)
  (define-elwise-node acosh)
  (define-elwise-node atanh)

  (define-elwise-node exp)
  (define-elwise-node log2)
  (define-elwise-node log10)
  (define-elwise-node logE))

;; ==============================================================
;; F() -> G(X, N, OUT) Function Family
;; ==============================================================

(defnode (ExptNode (myself)
	  :where `(X[~] OUT[~] N[scal] -> OUT[~] where scal = 1)
	  :documentation "The node ExptNode applies (expt N X) into each element, writing the result into out."))

(defun !expt (n x &key (out nil))
  "The function !expt applies (expt N X) into each element, writing the result into out.

Inputs:
    N - ScalarTensor
    X - AbstractTensor
    Out - AbstractTensor or nil

Output:
   out - AbstractTensor"
  (if out
      (forward (ExptNode) x out n)
      (forward (ExptNode) x (make-input (shape x) nil :dtype (dtype x) :order (order x)) n)))



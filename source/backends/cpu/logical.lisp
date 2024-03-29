 
(in-package :cl-waffe2/backends.cpu)

(eval-when (:compile-toplevel :load-toplevel :execute)  
  (defun where-simd-reject-p (&rest args &key &allow-other-keys)
    (let* ((compiler-info (car (last args))) ;; :compiler-info keyword
	   (op (car compiler-info)))
      (if *simd-extension-p*
	  (not
	   (or (eql op #'<)
	       (eql op #'<=)
	       (eql op #'>)
	       (eql op #'>=)	     
	       (eql op #'=)))
	  t)))

  (defun compare-simd-reject-p (&rest args)
    (let* ((op (car args)))
      (if *simd-extension-p*
	  (not
	   (or (eql op #'<)
	       (eql op #'<=)
	       (eql op #'>)
	       (eql op #'>=)	     
	       (eql op #'=)))
	  t))))

(defun cond->fname (dtype condition scal-p)
  (declare (type function condition))
  (cond
    ((eql condition #'<) (make-fname dtype "lt" :scal scal-p))
    ((eql condition #'<=) (make-fname dtype "le" :scal scal-p))
    ((eql condition #'>) (make-fname dtype "gt" :scal scal-p))
    ((eql condition #'>=) (make-fname dtype "ge" :scal scal-p))
    ((eql condition #'=) (make-fname dtype "eq" :scal scal-p))))

(define-impl-op (Where-Operation-Node
	      :device CPUTensor
	      :reject-p #'where-simd-reject-p)
	     :forward ((self x out)
		       (let* ((condition (car (logical-compiler-info self)))
			      (scal   (coerce (second (logical-compiler-info self)) (dtype->lisp-type (dtype x))))
			      (fname (symbol-function (cond->fname (dtype x) condition t)))
			      (then (logical-true-then self))
			      (else (logical-false-then self)))
			 (with-tensor-ptrs ((x-ptr x)
					    (o-ptr out))
			   (do-compiled-loop (list x out) ()
			       (x-view o-view)
			     (funcall
			      fname
			      (size-of x-view 0)
			      (incf-tensor-ptr x x-ptr :offset (offset-of x-view 0))
			      (stride-of x-view 0)
			      (incf-tensor-ptr out o-ptr :offset (offset-of o-view 0))
			      (stride-of o-view 0)
			      scal
			      then
			      else))
			   out))))

(define-impl-op (Compare-Operation-Node
		 :device CPUTensor
		 :reject-p #'compare-simd-reject-p)
		:forward ((self x y out)
			  (let* ((condition (logical-condition self))
				 (fname (symbol-function (cond->fname (dtype x) condition nil)))
				 (then (logical-true-then self))
				 (else (logical-false-then self)))
			    (with-tensor-ptrs ((x-ptr x)
					       (y-ptr y)
					       (o-ptr out))
			      (do-compiled-loop (list x y out) ()
				  (x-view y-view o-view)
				(funcall fname
					 (size-of x-view 0)
					 (incf-tensor-ptr x x-ptr :offset (offset-of x-view 0))
					 (stride-of x-view 0)
					 (incf-tensor-ptr y y-ptr :offset (offset-of y-view 0))
					 (stride-of y-view 0)
					 (incf-tensor-ptr out o-ptr :offset (offset-of o-view 0))
					 (stride-of o-view 0)
					 then else))
			      out))))



(in-package :cl-waffe2/backends.cpu)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

(macrolet ((define-math-impl (name)
	     (let ((node-name (symb name 'Node))
		   (sop-name   (symb 'waffe2-s name))
		   (dop-name   (symb 'waffe2-d name)))
	       `(define-impl (,node-name
			      :device CPUTensor
			      :reject-p #'simd-extension-p)
			     :forward ((self x out)
				       (let ((x-ptr (gensym "PTR"))
					     (o-ptr (gensym "PTR")))
					 `(locally (declare (optimize (speed 1)))
					    (with-tensor-ptrs ((,x-ptr ,x)
							       (,o-ptr ,out))
					      ,(call-with-view
						#'(lambda (x-view o-view)
						    `(,(case (dtype x)
							 (:float ',sop-name)
							 (:double ',dop-name)
							 (T (error "~a-CPUTensor has no implementation for ~a" ',node-name (dtype x))))
						      ,(size-of x-view 0)
						      (incf-tensor-ptr ,x ,x-ptr :offset ,(offset-of x-view 0))
						      ,(stride-of x-view 0)
						      (incf-tensor-ptr ,out ,o-ptr :offset ,(offset-of o-view 0))
						      ,(stride-of o-view 0)))
						`(,x ,out))
					      ,out))))))))					   
  (define-math-impl Sin)
  (define-math-impl Cos)
  (define-math-impl Tan)

  (define-math-impl ASin)
  (define-math-impl ACos)
  (define-math-impl ATan)

  (define-math-impl SinH)
  (define-math-impl CosH)
  (define-math-impl TanH)

  (define-math-impl Abs)
  (define-math-impl Exp)
  (define-math-impl LogE)
  (define-math-impl Log2)
  (define-math-impl Log10)
  (define-math-impl Log1p)

  (define-math-impl Sqrt))

(define-impl (ExptNode
	      :device CPUTensor
	      :cache-when-compiled nil
	      :reject-p #'simd-extension-p)
	     :forward ((self x out n)
		       (let ((x-ptr (gensym "PTR"))
			     (o-ptr (gensym "PTR"))
			     (n-ptr (gensym "PTR")))
			 `(locally (declare (optimize (speed 1)))
			    (with-tensor-ptrs ((,x-ptr ,x)
					       (,o-ptr ,out))
			      (let ((,n-ptr (coerce (tensor-vec ,n) ',(dtype->lisp-type (dtype x)))))
				,(call-with-view
				  #'(lambda (x-view o-view)
				      `(,(case (dtype x)
					   (:float 'waffe2-spow)
					   (:double 'waffe2-dpow)
					   (T (error "ExptNode-CPUTensor do not provide an implementation for ~a tensors" (dtype x))))
					,(size-of x-view 0)
					(incf-tensor-ptr ,x ,x-ptr :offset ,(offset-of x-view 0))
					,(stride-of x-view 0)
					,n-ptr
					(incf-tensor-ptr ,out ,o-ptr :offset ,(offset-of o-view 0))
					,(stride-of o-view 0)))
				  `(,x ,out))
				,out))))))

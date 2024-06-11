
(in-package :cl-waffe2/backends.cpu)

;; BLAS -> Lisp

(defun expand-axpy-form (x y x-ptr y-ptr &key (alpha 1.0))
  (let ((dtype (dtype x)))
    (call-with-view
     #'(lambda (x-view y-view)
	 ;; Adding Offsets...?
	 (case dtype
	   (:float
	    `(blas-saxpy
	      ,(size-of y-view 0)
	      ,alpha
	      (incf-tensor-ptr ,y ,y-ptr :offset ,(offset-of y-view 0))
	      ,(stride-of y-view 0)
	      (incf-tensor-ptr ,x ,x-ptr :offset ,(offset-of x-view 0))
	      ,(stride-of x-view 0)))
	   (:double
	    `(blas-daxpy
	      ,(size-of y-view 0)
	      ,(coerce alpha 'double-float)
	      (incf-tensor-ptr ,y ,y-ptr :offset ,(offset-of y-view 0))
	      ,(stride-of y-view 0)
	      (incf-tensor-ptr ,x ,x-ptr :offset ,(offset-of x-view 0))
	      ,(stride-of x-view 0)))
	   (T
	    (error "the dtype ~a is not supported. (TODO)" dtype))))
     `(,x ,y))))

(defun expand-arithmetic-form (x y x-ptr y-ptr &key (fname "add"))
  (let ((fname (make-fname (dtype x) fname)))
    (call-with-view
     #'(lambda (x-view y-view)
	 `(,fname ,(size-of y-view 0) (incf-tensor-ptr ,x ,x-ptr :offset ,(offset-of x-view 0)) ,(stride-of x-view 0) (incf-tensor-ptr ,y ,y-ptr :offset ,(offset-of y-view 0)) ,(stride-of y-view 0)))
     `(,x ,y))))

(defun expand-move-form (x y x-ptr y-ptr)
  (let ((dtype (dtype x)))
    (call-with-view
     #'(lambda (x-view y-view)
	 ;; Adding Offsets...?
	 (case dtype
	   (:float
	    `(blas-scopy
	      ,(size-of y-view 0)
	      (incf-tensor-ptr ,y ,y-ptr :offset ,(offset-of y-view 0))
	      ,(stride-of y-view 0)
	      (incf-tensor-ptr ,x ,x-ptr :offset ,(offset-of x-view 0))
	      ,(stride-of x-view 0)))
	   (:double
	    `(blas-dcopy
	      ,(size-of y-view 0)
	      (incf-tensor-ptr ,y ,y-ptr :offset ,(offset-of y-view 0))
	      ,(stride-of y-view 0)
	      (incf-tensor-ptr ,x ,x-ptr :offset ,(offset-of x-view 0))
	      ,(stride-of x-view 0)))
	   (T
	    (error "the dtype ~a is not supported. (TODO)" dtype))))
     `(,x ,y))))

;; reject-p return t to ignore the dispatching

(define-impl (AddNode :device CPUTensor
		      :reject-p #'(lambda (dtype)
				    (if (or (eql dtype :float)
					    (eql dtype :double))
					nil
					(not *simd-extension-p*)))) ;; If simd-extension has loaded, it can handle with sparse matrix.
	     :forward ((self x y)
		       (let ((x-ptr (gensym "PTR"))
			     (y-ptr (gensym "PTR")))
			 `(with-tensor-ptrs ((,x-ptr ,x)
					     (,y-ptr ,y))
			    (locally (declare (optimize (speed 1)))
			      ,(if *simd-extension-p*
				   (expand-arithmetic-form  x y x-ptr y-ptr :fname "add")
				   (expand-axpy-form x y x-ptr y-ptr))
			      ,x)))))

(define-impl (SubNode :device CPUTensor
		      :reject-p #'(lambda (dtype)
				    (if (or (eql dtype :float)
					    (eql dtype :double))
					nil
					(not *simd-extension-p*))))
	     :forward ((self x y)
		       (let ((x-ptr (gensym "PTR"))
			     (y-ptr (gensym "PTR")))
			 `(with-tensor-ptrs ((,x-ptr ,x)
					     (,y-ptr ,y))
			    (locally (declare (optimize (speed 1)))			      
			      ,(if *simd-extension-p*
				   (expand-arithmetic-form x y x-ptr y-ptr :fname "sub")
				   (expand-axpy-form x y x-ptr y-ptr :alpha -1.0))
			      ,x)))))

(define-impl (MulNode :device CPUTensor
		      :reject-p #'(lambda (dtype)
				    (declare (ignore dtype))
				    (not *simd-extension-p*)))
	     :forward ((self x y)
		       (let ((x-ptr (gensym "PTR"))
			     (y-ptr (gensym "PTR")))
			 `(with-tensor-ptrs ((,x-ptr ,x)
					     (,y-ptr ,y))
			    (locally (declare (optimize (speed 1)))			      
			      ,(expand-arithmetic-form x y x-ptr y-ptr :fname "mul")
			      ,x)))))

(define-impl (DivNode :device CPUTensor
		      :reject-p #'(lambda (dtype)
				    (declare (ignore dtype))
				    (not *simd-extension-p*)))
	     :forward ((self x y)
		       (let ((x-ptr (gensym "PTR"))
			     (y-ptr (gensym "PTR")))
			 `(with-tensor-ptrs ((,x-ptr ,x)
					     (,y-ptr ,y))
			    (locally (declare (optimize (speed 1)))			      
			      ,(expand-arithmetic-form x y x-ptr y-ptr :fname "div")
			      ,x)))))

(define-impl (MoveTensorNode :device CPUTensor
			     :reject-p #'(lambda (dtype &rest more-args)
					   (declare (ignore more-args))
					   (if (or (eql dtype :float)
						   (eql dtype :double))
					       nil
					       (not *simd-extension-p*))))
	     :forward ((self x y)
		       ;; X <- Y
		       (let ((x-ptr (gensym "PTR"))
			     (y-ptr (gensym "PTR")))
			 `(with-tensor-ptrs ((,x-ptr ,x)
					     (,y-ptr ,y))
			    (locally (declare (optimize (speed 1)))
			      ,(if (or (eql (dtype x) :float)
				       (eql (dtype x) :double))
				   (expand-move-form x y x-ptr y-ptr)					 
				   ;; *simd-extension-p*=t
				   (expand-arithmetic-form x y x-ptr y-ptr :fname "copy"))
			      ,x)))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun simd-extension-p (&rest args)
    (declare (ignore args))
    (not *simd-extension-p*)))


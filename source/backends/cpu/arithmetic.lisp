
(in-package :cl-waffe2/backends.cpu)


(defun expand-axpy-form (x y &key (alpha 1.0))
  (let ((dtype (dtype x)))
    (call-with-view
     #'(lambda (x-view y-view)
	 ;; Adding Offsets...?
	 (case dtype
	   (:float
	    `(blas-saxpy
	      ,(size-of x-view 0)
	      ,alpha
	      ,y
	      ,(stride-of y-view 0)
	      ,x
	      ,(stride-of x-view 0)))
	   (:double
	    `(blas-daxpy
	      ,(size-of x-view 0)
	      ,alpha
	      ,y
	      ,(stride-of y-view 0)
	      ,x
	      ,(stride-of x-view 0)))
	   (T
	    (error "the dtype ~a is not supported. (TODO)" dtype))))
     `(,x ,y))))

(define-impl (AddNode :device CPUTensor)
	     :forward ((self x y)
		       `(,@(expand-axpy-form x y)
			 ,x))
	     :backward ((self dout dx dy)
			(values
			 (!move dx dout)
			 (!move dy dout))))

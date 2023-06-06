
(in-package :cl-waffe2/backends.cpu)


(defun expand-axpy-form (x y &key (alpha 1.0))
  (let ((dtype (dtype x)))
    (call-with-view
     #'(lambda (x-view y-view)
	 ;; Adding Offsets...?
	 (case dtype
	   (:float
	    `(blas-saxpy
	      ,(size-of y-view 0)
	      ,alpha
	      (tensor-ptr ,y :offset ,(offset-of y-view 0))
	      ,(stride-of y-view 0)
	      (tensor-ptr ,x :offset ,(offset-of x-view 0))
	      ,(stride-of x-view 0)))
	   (:double
	    `(blas-daxpy
	      ,(size-of y-view 0)
	      ,alpha
	      (tensor-ptr ,y :offset ,(offset-of y-view 0))
	      ,(stride-of y-view 0)
	      (tensor-ptr ,x :offset ,(offset-of x-view 0))
	      ,(stride-of x-view 0)))
	   (T
	    (error "the dtype ~a is not supported. (TODO)" dtype))))
     `(,x ,y))))

(defun expand-move-form (x y)
  (let ((dtype (dtype x)))
    (call-with-view
     #'(lambda (x-view y-view)
	 ;; Adding Offsets...?
	 (case dtype
	   (:float
	    `(blas-scopy
	      ,(size-of y-view 0)
	      (tensor-ptr ,y :offset ,(offset-of y-view 0))
	      ,(stride-of y-view 0)
	      (tensor-ptr ,x :offset ,(offset-of x-view 0))
	      ,(stride-of x-view 0)))
	   (:double
	    `(blas-dcopy
	      ,(size-of y-view 0)
	      (tensor-ptr ,y :offset ,(offset-of y-view 0))
	      ,(stride-of y-view 0)
	      (tensor-ptr ,x :offset ,(offset-of x-view 0))
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

(define-impl (SubNode :device CPUTensor)
	     :forward ((self x y)
		       `(,@(expand-axpy-form x y :alpha -1.0)
			 ,x))
	     :backward ((self dout dx dy)
			(values
			 (!move dx dout)
			 (!move dy (!mul -1.0 dout)))))

(define-impl (MoveTensorNode :device CPUTensor)
	     :forward ((self x y)
		       ;; X <- Y
		       `(,@(expand-move-form x y)
			 ,x))
	     :backward ((self dout dx dy)
			(declare (ignore dx))
			(let ((dy-out
				(if (and
				     (eql (tensor-attribute dy) :chain)
				     (movetensor-ignore-me self))
				    dout
				    (!copy dout))))
			  (values dout dy-out))))


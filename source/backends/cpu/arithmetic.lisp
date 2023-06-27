
(in-package :cl-waffe2/backends.cpu)

;; BLAS -> Lisp

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
	      ,(coerce alpha 'double-float)
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

(define-impl (AddNode :device CPUTensor
	      :reject-p (supported-dtypes-are 0 :float :double))
	     :forward ((self x y)
		       `(,@(expand-axpy-form x y)
			 ,x)))

(define-impl (SubNode :device CPUTensor
	      :reject-p (supported-dtypes-are 0 :float :double))
	     :forward ((self x y)
		       `(,@(expand-axpy-form x y :alpha -1.0)
			 ,x)))

;; MulNode/DivNode -> LispKernel
(define-impl (MoveTensorNode :device CPUTensor
			     :reject-p (supported-dtypes-are 0 :float :double))
	     :forward ((self x y)
		       ;; X <- Y
		       `(progn
			  (if (not (movetensor-ignore-me ,self))
			      (progn
				,(expand-move-form x y)
				,x)
			      ,y))))


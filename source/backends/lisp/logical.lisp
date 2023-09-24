
(in-package :cl-waffe2/backends.lisp)


;; TODO: should be optimized (is the function vectorized?)
(define-with-typevar (where-kernel u) (x y
				       cond true-then false-then
				       size offsetx offsety incx incy)
  (declare (optimize (speed 3) (safety 1))
	   (type (simple-array u (*)) x y)
	   (type fixnum offsetx offsety incx incy size)
	   (type function cond))

  (dotimes (i size)
    (setf (aref y (+ offsety (the fixnum (* incy i))))
	  (if (funcall cond (aref x (+ offsetx (the fixnum (* incx i)))))
	      true-then
	      false-then))))

;; TODO: Use AVX2? (should be realised on other backends?)
(define-with-typevar (compare-kernel u) (x y out
					 cond
					 true-then false-then
					 size
					 offsetx offsety offseto
					 incx incy inco)
  (declare (optimize (speed 3) (safety 1))
	   (type (simple-array u (*)) x y out)
	   (type fixnum offsetx offsety offseto incx incy inco size)
	   (type function cond))

  (dotimes (i size)
    (setf (aref out (+ offseto (the fixnum (* inco i))))
	  (if (funcall cond
		       (aref x (+ offsetx (the fixnum (* incx i))))
		       (aref y (+ offsety (the fixnum (* incy i)))))
	      true-then
	      false-then))))

(define-impl-op (Where-Operation-Node :device LispTensor)
		:forward ((self tensor out)
			  (let ((kernel (where-kernel (dtype tensor))))
			    (do-compiled-loop (list tensor out) ()
				(x-view o-view)
			      (funcall kernel
				       (tensor-vec tensor)
				       (tensor-vec out)
				       (logical-condition self)
				       (logical-true-then self)
				       (logical-false-then self)
				       (size-of x-view 0)
				       (offset-of x-view 0)
				       (offset-of o-view 0)
				       (stride-of x-view 0)
				       (stride-of o-view 0)))
			    out)))

(define-impl-op (Compare-Operation-Node :device LispTensor)
		:forward ((self tensor1 tensor2 out)
			  (let ((kernel (compare-kernel (dtype tensor1))))
			    (do-compiled-loop (list tensor1 tensor2 out) ()
				(x-view y-view o-view)
			      (funcall kernel
				       (tensor-vec tensor1)
				       (tensor-vec tensor2)
				       (tensor-vec out)
				       (logical-condition self)
				       (logical-true-then self)
				       (logical-false-then self)
				       (size-of x-view 0)
				       (offset-of x-view 0)
				       (offset-of y-view 0)
				       (offset-of o-view 0)
				       (stride-of x-view 0)
				       (stride-of y-view 0)
				       (stride-of o-view 0)))
			    (print out)
			    out)))

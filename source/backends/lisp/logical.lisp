
(in-package :cl-waffe2/backends.lisp)


;; TODO: should be optimized (is the function vectorized?)
(define-with-typevar (where-kernel u) (x y
				       cond true-then false-then
				       size offsetx offsety incx incy)
  (declare (optimize (speed 3) (safety 0))
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
  (declare (optimize (speed 3) (safety 0))
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

(defun expand-where-form (node tensor out)
  `(,@(call-with-view
       #'(lambda (x-view y-view)
	   (let ((size (size-of x-view 0))
		 (incx (stride-of x-view 0))
		 (incy (stride-of y-view 0))
		 (offsetx    (offset-of x-view 0))
		 (offsety    (offset-of y-view 0))
		 (condition  (logical-condition node))
		 (true-case  (logical-true-then node))
		 (false-case (logical-false-then node))
		 (kernel (where-kernel (dtype tensor))))
	     `(funcall
	       ,kernel
	       (tensor-vec ,tensor)
	       (tensor-vec ,out)
	       ,condition
	       ,true-case
	       ,false-case
	       ,size
	       ,offsetx
	       ,offsety
	       ,incx
	       ,incy)))
       `(,tensor ,out))))

(defun expand-compare-form (node tensor1 tensor2 out)
  `(,@(call-with-view
       #'(lambda (x-view y-view o-view)
	   (let ((size (size-of x-view 0))
		 (incx (stride-of x-view 0))
		 (incy (stride-of y-view 0))
		 (inco (stride-of o-view 0))
		 (offsetx (offset-of x-view 0))
		 (offsety (offset-of y-view 0))
		 (offseto (offset-of o-view 0))
		 (condition  (logical-condition node))
		 (true-case  (logical-true-then node))
		 (false-case (logical-false-then node))
		 (kernel (compare-kernel (dtype tensor1))))
	     `(funcall ,kernel
		       (tensor-vec ,tensor1)
		       (tensor-vec ,tensor2)
		       (tensor-vec ,out)
		       ,condition
		       ,true-case ,false-case
		       ,size
		       ,offsetx ,offsety ,offseto
		       ,incx ,incy ,inco)))
       `(,tensor1 ,tensor2 ,out))))


(define-impl (Where-Operation-Node :device LispTensor)
	     :forward ((self tensor out)
		       `(,@(expand-where-form self tensor out)
			  ,out)))

(define-impl (Compare-Operation-Node :device LispTensor)
	     :forward ((self tensor1 tensor2 out)
		       `(progn
			  ,(expand-compare-form self tensor1 tensor2 out)
			  ,out)))

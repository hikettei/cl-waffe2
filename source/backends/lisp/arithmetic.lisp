
(in-package :cl-waffe2/backends.lisp)

(macrolet ((define-arith-func (name f)
	     `(define-with-typevar (,name u) (x y offsetx offsety size incx incy)
		(declare (optimize (speed 3))
			 (type (simple-array u (*)) x y)
			 (type fixnum offsetx offsety size incx incy))
		(dotimes (i size)
		  (setf (aref x (+ offsetx (the fixnum (* incx i))))
			(,f (aref x (+ offsetx (the fixnum (* incx i))))
			    (aref y (+ offsety (the fixnum (* incy i))))))))))
  (define-arith-func matrix-add +)
  (define-arith-func matrix-sub -)
  (define-arith-func matrix-mul *)
  (define-arith-func matrix-div /))

(macrolet ((define-scalar-func (name f)
	     `(define-with-typevar (,name u) (x scalar offsetx size incx)
		(declare (optimize (speed 3))
			 (type (simple-array u (*)) x)
			 (type u scalar)
			 (type fixnum offsetx size incx))
		(dotimes (i size)
		  (setf (aref x (+ offsetx (the fixnum (* incx i))))
			(,f (aref x (+ offsetx (the fixnum (* incx i))))
			    scalar))))))
  (define-scalar-func scalar-add +)
  (define-scalar-func scalar-sub -)
  (define-scalar-func scalar-mul *)
  (define-scalar-func scalar-div /))

(define-with-typevar (matrix-inv u) (x offsetx size incx)
  (declare (optimize (speed 3))
	   (type (simple-array u (*)) x)
	   (type fixnum offsetx size incx))
  
  (dotimes (i size)
    (setf (aref x (+ offsetx (the fixnum (* incx i))))
	  (/ 1 (aref x (+ offsetx (the fixnum (* incx i))))))))

;; Even on SBCL:
;; (disassemble (add-matrix :uint8)) <- Fails to SIMDify
;; (disassemble (add-matrix :float)) <- using addss (AVX2 Only?)
;; (disassemble (add-matrix :double)) <- using addsd (AVX2 Only?)

;; define them with macrolet.

(define-impl (AddNode :device LispTensor)
	     :forward ((self x y)
		       (let ((adder (matrix-add (dtype x))))
			 `(,@(call-with-view
			      #'(lambda (x-view
					 y-view)
				  `(funcall ,adder
			 		    (tensor-vec ,x)
					    (tensor-vec ,y)
					    ,(offset-of x-view 0)
					    ,(offset-of y-view 0)
					    ,(size-of x-view 0)
					    ,(stride-of x-view 0)
					    ,(stride-of y-view 0)))
			      `(,x ,y))
			   ,x))))

(define-impl (SubNode :device LispTensor)
	     :forward ((self x y)
		       (let ((subber (matrix-sub (dtype x))))
			 `(,@(call-with-view
			      #'(lambda (x-view
					 y-view)
				  `(funcall ,subber
			 		    (tensor-vec ,x)
					    (tensor-vec ,y)
					    ,(offset-of x-view 0)
					    ,(offset-of y-view 0)
					    ,(size-of x-view 0)
					    ,(stride-of x-view 0)
					    ,(stride-of y-view 0)))
			      `(,x ,y))
			   ,x))))


(define-impl (MulNode :device LispTensor)
	     :save-for-backward (t t)
	     :forward ((self x y)
		       (let ((multiplier (matrix-mul (dtype x))))
			 `(,@(call-with-view
			      #'(lambda (x-view
					 y-view)
				  `(funcall ,multiplier
			 		    (tensor-vec ,x)
					    (tensor-vec ,y)
					    ,(offset-of x-view 0)
					    ,(offset-of y-view 0)
					    ,(size-of x-view 0)
					    ,(stride-of x-view 0)
					    ,(stride-of y-view 0)))
			      `(,x ,y))
			   ,x))))


(define-impl (DivNode :device LispTensor)
	     :save-for-backward (t t)
	     :forward ((self x y)
		       (let ((divider (matrix-div (dtype x))))
			 `(,@(call-with-view
			      #'(lambda (x-view
					 y-view)
				  `(funcall ,divider
			 		    (tensor-vec ,x)
					    (tensor-vec ,y)
					    ,(offset-of x-view 0)
					    ,(offset-of y-view 0)
					    ,(size-of x-view 0)
					    ,(stride-of x-view 0)
					    ,(stride-of y-view 0)))
			      `(,x ,y))
			   ,x))))


(define-impl (InverseTensorNode :device LispTensor)
	     :save-for-backward (t)
	     :forward
	     ((self x)
	      (let ((inver (matrix-inv (dtype x))))
		`(,@(call-with-view
		     #'(lambda (x-view)
			 `(funcall
			   ,inver
			   (tensor-vec ,x)
			   ,(offset-of x-view 0)
			   ,(size-of x-view 0)
			   ,(stride-of x-view 0)))
		     `(,x))
		  ,x))))

(define-impl (ScalarAdd :device LispTensor)
	     :forward
	     ((self x scalar)
	      (let ((adder (scalar-add (dtype x))))
		`(,@(call-with-view
		     #'(lambda (x-view)
			 `(funcall
			   ,adder
			   (tensor-vec ,x)
			   (tensor-vec ,scalar)
			   ,(offset-of x-view 0)
			   ,(size-of x-view 0)
			   ,(stride-of x-view 0)))
		     `(,x))
		  ,x))))

(define-impl (ScalarSub :device LispTensor)
	     :forward
	     ((self x scalar)
	      (let ((subber (scalar-sub (dtype x))))
		`(,@(call-with-view
		     #'(lambda (x-view)
			 `(funcall
			   ,subber
			   (tensor-vec ,x)
			   (tensor-vec ,scalar)
			   ,(offset-of x-view 0)
			   ,(size-of x-view 0)
			   ,(stride-of x-view 0)))
		     `(,x))
		  ,x))))


(define-impl (ScalarMul :device LispTensor)
	     :save-for-backward (t t)
	     :forward
	     ((self x scalar)
	      (let ((multiplier (scalar-mul (dtype x))))
		`(,@(call-with-view
		     #'(lambda (x-view)
			 `(funcall
			   ,multiplier
			   (tensor-vec ,x)
			   (tensor-vec ,scalar)
			   ,(offset-of x-view 0)
			   ,(size-of x-view 0)
			   ,(stride-of x-view 0)))
		     `(,x))
		  ,x))))

(define-impl (ScalarDiv :device LispTensor)
	     :save-for-backward (t t)
	     :forward
	     ((self x scalar)
	      (let ((divider (scalar-div (dtype x))))
		`(,@(call-with-view
		     #'(lambda (x-view)
			 `(funcall
			   ,divider
			   (tensor-vec ,x)
			   (tensor-vec ,scalar)
			   ,(offset-of x-view 0)
			   ,(size-of x-view 0)
			   ,(stride-of x-view 0)))
		     `(,x))
		  ,x))))

(define-with-typevar (matrix-move u) (out x offseto offsetx inco incx size)
  (declare (optimize (speed 3))
	   (type (simple-array u (*)) x out)
	   (type fixnum offseto offsetx inco incx size))
  (dotimes (i size)
    ;; out <- x
    (setf (aref out (+ offseto (the fixnum (* inco i))))
	  (aref x   (+ offsetx (the fixnum (* incx i)))))))

(defun total-of (tensor)
  (apply #'* (shape tensor)))

(define-impl (MoveTensorNode :device LispTensor)
	     :forward
	     ((self x y)
	      ;; x <- y.
	      ;; place val
	      (let ((mover (matrix-move (dtype x))))
		`(progn
		   ,(call-with-view
		     #'(lambda (x-view y-view)
			 `(funcall
			   ,mover
			   (tensor-vec ,x)
			   (tensor-vec ,y)
			   ,(offset-of x-view 0)
			   ,(offset-of y-view 0)
			   ,(stride-of x-view 0)
			   ,(stride-of y-view 0)
			   ,(size-of x-view 0)))
		     `(,x ,y))
		   ,x))))


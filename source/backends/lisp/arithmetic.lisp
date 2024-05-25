
(in-package :cl-waffe2/backends.lisp)

;; [TODO] Replace them with iterator
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

;; Even on SBCL:
;; (disassemble (add-matrix :uint8)) <- Fails to SIMDify
;; (disassemble (add-matrix :float)) <- using addss (AVX2 Only?)
;; (disassemble (add-matrix :double)) <- using addsd (AVX2 Only?)

;; define them with macrolet.

(define-impl (AddNode :device LispTensor)
	     :forward ((self x y)
		       (let ((adder (matrix-add (dtype x))))
			 ;; iteration.lisp can be by far faster altenative of call-with-view
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


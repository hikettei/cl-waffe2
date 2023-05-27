
(in-package :cl-waffe2/backends.lisp)

(macrolet ((define-arith-func (name f)
	     `(define-with-typevar (,name u) (x y offsetx offsety size)
		(declare (optimize (speed 3))
			 (type (simple-array u (*)) x y)
			 (type fixnum offsetx offsety size))
		(dotimes (i size)
		  (setf (aref x (+ offsetx i))
			(,f (aref x (+ offsetx i))
			    (aref y (+ offsety i))))))))
  (define-arith-func matrix-add +)
  (define-arith-func matrix-sub -)
  (define-arith-func matrix-mul *)
  (define-arith-func matrix-div /))

;; (disassemble (add-matrix :uint8))
	     
(define-impl (AddNode :device LispTensor)
	     :forward ((self x y)
		       (let ((adder (matrix-add (dtype x))))
		       `(,@(call-with-view
			    #'(lambda (x-view
				       y-view)
				`(funcall ,adder
			 	  (tensor-vec ,x)
				  (tensor-vec ,y)
				  ,(viewinstruction-offset x-view)
				  ,(viewinstruction-offset y-view)
				  ,(viewinstruction-size x-view)))
			    `(,x ,y))
			 ,x)))
	     :backward ((self dy)
			`(values ,dy ,dy)))

	     
(define-impl (SubNode :device LispTensor)
	     :forward ((self x y)
		       (let ((subber (matrix-sub (dtype x))))
		       `(,@(call-with-view
			    #'(lambda (x-view
				       y-view)
				`(funcall ,subber
			 	  (tensor-vec ,x)
				  (tensor-vec ,y)
				  ,(viewinstruction-offset x-view)
				  ,(viewinstruction-offset y-view)
				  ,(viewinstruction-size x-view)))
			    `(,x ,y))
			 ,x)))
	     :backward ((self dy)
			`(values ,dy ,dy)))


(define-impl (MulNode :device LispTensor)
	     :forward ((self x y)
		       (let ((multiplier (matrix-mul (dtype x))))
		       `(,@(call-with-view
			    #'(lambda (x-view
				       y-view)
				`(funcall ,multiplier
			 	  (tensor-vec ,x)
				  (tensor-vec ,y)
				  ,(viewinstruction-offset x-view)
				  ,(viewinstruction-offset y-view)
				  ,(viewinstruction-size x-view)))
			    `(,x ,y))
			 ,x)))
	     :backward ((self dy)
			`(values ,dy ,dy)))

(define-impl (DivNode :device LispTensor)
	     :forward ((self x y)
		       (let ((divider (matrix-div (dtype x))))
		       `(,@(call-with-view
			    #'(lambda (x-view
				       y-view)
				`(funcall ,divider
			 	  (tensor-vec ,x)
				  (tensor-vec ,y)
				  ,(viewinstruction-offset x-view)
				  ,(viewinstruction-offset y-view)
				  ,(viewinstruction-size x-view)))
			    `(,x ,y))
			 ,x)))
	     :backward ((self dy)
			`(values ,dy ,dy)))





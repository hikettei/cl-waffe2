
(in-package :cl-waffe2/distributions)

;; Xavier/Xe/He, Orthogonal.

(define-with-typevar (simple-array-sample! u) (size array function)
  (declare (optimize (speed 3))
	   (type (simple-array u (*)) array)
	   (type fixnum size)
	   (type function function))
  (dotimes (i size)
    (setf (aref array i) (funcall function i))))

(defun initialize-vec! (tensor function)
  "
function ... lambda(i) where i = index"
  (declare (type AbstractTensor tensor)
	   (type function function))

  ;; TODO: Coerce into their type.
  ;; TODO: double
  (maybe-with-lparallel
    (typecase (tensor-vec tensor)
      (simple-array
       (maybe-pfuncall (simple-array-sample! (dtype tensor))
		       (apply #'* (shape tensor))
		       (tensor-vec tensor)
		       function))
      (T
       (loop for i fixnum
	     upfrom 0
	       below (apply #'* (shape tensor))
	     do (setf (vref (the simple-array (tensor-vec tensor)) i) (funcall function i)))))))

;; TODO: AddDoc distribution sampler.
(macrolet ((define-initializer-function (function-name
					 (&rest args)
					 initializer-lambda
					 document)
	     `(progn
		(export ',function-name)
		(defun ,function-name (shape-or-scalar ,@args &rest initargs &key &allow-other-keys)
		  ,document
		  (let ((tensor (apply #'make-tensor shape-or-scalar initargs)))
		    (initialize-vec! tensor ,initializer-lambda)
		    tensor)))))
  
  (define-initializer-function
      uniform-random
      (upper)
    (let ((upper (coerce upper (dtype->lisp-type (dtype tensor)))))
      #'(lambda (i)
	  (declare (ignore i))
	  (random upper)))
    "The function uniform-random samples the uniform-random from the function (random n), returning the new tensor with the given shape.")
  
  (define-initializer-function
      ax+b
      (a b)
    (let ((a (coerce a (dtype->lisp-type (dtype tensor))))
	  (b (coerce b (dtype->lisp-type (dtype tensor)))))
      #'(lambda (i) (step-ax+b a i b)))
    "The function ax+b samples the new tensor following this sequence:
Tensor[Index] = a*Index + b")

  (define-initializer-function
      beta
      (alpha beta)
    (let* ((a (coerce alpha (dtype->lisp-type (dtype tensor))))
	   (b (coerce beta  (dtype->lisp-type (dtype tensor))))
	   (a (max a b))
	   (b (min a b))
	   (sampler (if (> a 1.0)
			(beta-bb (dtype tensor))
			(beta-bc (dtype tensor)))))
      #'(lambda (i)
	  (declare (ignore i))
	  (funcall sampler alpha a b)))
    "The function beta samples beta distributions using this algorithm: https://dl.acm.org/doi/pdf/10.1145/359460.359482")

  (define-initializer-function
      randn
      ()
    (let* ((sampler (get-randn-sampler (dtype tensor))))
      #'(lambda (i)
	  (declare (ignore i))
	  (funcall sampler)))
    "The function randn samples the gaussian distributions using ziggurat algorithm.")

  )



(in-package :cl-waffe2/distributions)

;; TODO: Optimize
(defun initialize-vec! (tensor function)
  "
function ... lambda(i) where i = index"
  (declare (type AbstractTensor tensor)
	   (type function function))

  (typecase (tensor-vec tensor)
    (simple-array
     (loop for i fixnum
	   upfrom 0
	     below (apply #'* (shape tensor))
	   do (setf (aref (the simple-array (tensor-vec tensor)) i) (funcall function i))))
    (T
     (loop for i fixnum
	   upfrom 0
	     below (apply #'* (shape tensor))
	   do (setf (vref (the simple-array (tensor-vec tensor)) i) (funcall function i))))))

(macrolet ((define-initializer-function (function-name
					 (&rest args)
					 initializer-lambda
					 document
					 &aux (tensor (gensym)))
	     `(progn
		(export ',function-name)
		(defun ,function-name (shape-or-scalar ,@args &rest initargs &key &allow-other-keys)
		  ,document
		  (let ((,tensor (apply #'make-tensor shape-or-scalar initargs)))
		    (initialize-vec! ,tensor ,initializer-lambda)
		    ,tensor)))))
  
  (define-initializer-function
      uniform-random
      (upper)
    #'(lambda (i) (declare (ignore i)) (random upper))
    "")

  )

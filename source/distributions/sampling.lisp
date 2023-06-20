
(in-package :cl-waffe2/distributions)

;; Xavier/Xe/He, Orthogonal.

(define-with-typevar (simple-array-sample! u) (size array function)
  (declare (optimize (speed 3))
	   (type (simple-array u (*)) array)
	   (type fixnum size)
	   (type function function))
  (dotimes (i size)
    (setf (aref array i) (funcall function i))))

(defun initialize-vec! (tensor
			function
			&key (keep-order? nil))
  "
function ... lambda(i) where i = index
If keep-order = t, forcibly it uses mref (with computing strides). This option is important for fast sampling distributions.
"
  (declare (type AbstractTensor tensor)
	   (type function function))

  (maybe-with-lparallel
    (cond
      ((and (typep (tensor-vec tensor) 'simple-array)
	    (not keep-order?))
       ;; The fastest case: the type of tensor-vec is already known.
       ;; Continue with ignoring strides
       (funcall (simple-array-sample! (dtype tensor))
		(apply #'* (shape tensor))
		(tensor-vec tensor)
	        function))
      ((or (not keep-order?) (eql (order tensor) :column))
       ;; The second fastest case: the type of tensor-vec isn't known but we can access it element by element each other.
       ;; Continue with ignoring strides
       (loop for i fixnum
	     upfrom 0
	       below (apply #'* (shape tensor))
	     do (setf (vref tensor i) (maybe-pfuncall function i))))
      (T
       ;; The slowest case: We have to access tensor-vec's element with considering strides...

       (let ((count 0))
	 (labels ((explore (rest-dim subscripts)
		    (if (= rest-dim (1- (length (shape tensor))))
			(dotimes (i (nth rest-dim (shape tensor)))
			  (apply
			   #'(setf mref)
			   (maybe-pfuncall function count)
			   tensor
			   `(,@subscripts ,i))
			  (incf count))
			(dotimes (i (nth rest-dim (shape tensor)))
			  (explore
			   (1+ rest-dim)
			   `(,@subscripts ,i))))))
	   (explore 0 nil)))))))

;; TODO: AddDoc distribution sampler.
;; BASIC Format: (distribution-name shape (If Any, arguments for distribution) &rest make-tensor's keywords...)

;; Export this macro as user-extensible macro at other name.?
;; TODO: Optimize -> coerce
(macrolet ((define-initializer-function (function-name
					 (&rest args)
					 initializer-lambda
					 document
					 &optional keep-order?)
	     ;; Set t if use the position of element in the tensor.
	     `(progn
		(export ',function-name)
		(defun ,function-name (shape ,@args &rest initargs &key &allow-other-keys)
		  ,document
		  (let ((tensor (apply #'make-tensor shape initargs)))
		    (initialize-vec! tensor ,initializer-lambda :keep-order? ,keep-order?)
		    tensor)))))
  
  (define-initializer-function
      uniform-random
      (upfrom below)
    (let ((upfrom (coerce upfrom (dtype->lisp-type (dtype tensor))))
	  (below  (coerce below  (dtype->lisp-type (dtype tensor)))))
      #'(lambda (i)
	  (declare (ignore i))
	  (sample-uniform-random upfrom below)))
    "The function uniform-random samples the uniform-random from the function (random n), returning the new tensor with the given shape.

Each element, x = [upfrom, below)")
  
  (define-initializer-function
      ax+b
      (a b)
    (let ((a (coerce a (dtype->lisp-type (dtype tensor))))
	  (b (coerce b (dtype->lisp-type (dtype tensor)))))
      #'(lambda (i) (step-ax+b a i b)))
    "The function ax+b samples the new tensor following this sequence:
Tensor[Index] = a*Index + b"
    t)

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
      normal
      (mean stddev)
    (let* ((mean (coerce mean 'double-float))
	   (stddev (coerce stddev 'double-float)))
      #'(lambda (i)
	  (declare (ignore i))
	  (coerce (cl-randist:random-normal-ziggurat mean stddev) (dtype->lisp-type (dtype tensor)))))
    "normal dist")

  (define-initializer-function
      randn
      ()
    (let* ((sampler (get-randn-sampler (dtype tensor))))
      #'(lambda (i)
	  (declare (ignore i))
	  (funcall sampler)))
    "The function randn samples the gaussian distributions using ziggurat algorithm.")

  (define-initializer-function
      expotential
      ()
    (let ((sampler (get-expotential-sampler (dtype tensor))))
      #'(lambda (i)
	  (declare (ignore i))
	  (funcall sampler)))
    "The function expotential samples the expotential distribution using ziggurat algorithm.")

  (define-initializer-function
      gamma
      (k)
    #'(lambda (i)
	(declare (ignore i))
	(sample-gamma (coerce k (dtype->lisp-type (dtype tensor)))))
    "The function gamma samples the gamma distributions.")

  (define-initializer-function
      bernoulli
      (p)
    #'(lambda (i)
	(declare (ignore i))
	(coerce (sample-bernoulli p) (dtype->lisp-type (dtype tensor))))
    "The bernoulli samples the bernoulli distributions.")

  (define-initializer-function
      chisquare
      (df)
    #'(lambda (i)
	(declare (ignore i))
	(sample-gamma (coerce (/ df 2.0) (dtype->lisp-type (dtype tensor)))))
    "The function chisquare samples the chisquare distribution."))


(defmacro define-tensor-initializer (function-name
				     (&rest args)
				     initializer-lambda
				     document
				     &optional keep-order?)
  "defines a initializer function

keep-order? set t if you're going to using the position of element."
  `(defun ,function-name (shape ,@args &rest initargs &key &allow-other-keys)
     ,document
     (let ((tensor (apply #'make-tensor shape initargs)))
       (initialize-vec! tensor ,initializer-lambda :keep-order? ,keep-order?)
       tensor)))


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

;; TO ADD:
;;
;; Orthogonal
;; Xe/He/Xavier
;;

;; TODO: Optimize -> coerce
(macrolet ((define-initializer-function (function-name
					 (&rest args)
					 initializer-lambda
					 document
					 &optional keep-order?)
	     ;; Set t if use the position of element in the tensor.
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',function-name)
		(defun ,function-name (shape ,@args &rest initargs &key &allow-other-keys)
		  ,(format nil "
```lisp
(~(~a~) shape ~(~a~)&rest initargs &key &allow-other-keys)
```
~a"
			   (symbol-name function-name)
			   (with-output-to-string (o)
			     (dolist (arg args) (princ (symbol-name arg) o) (princ " " o)))
			   document)
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
    "The function uniform-random is a family of initializer functions, and samples matrices from uniform random distribution using Common Lisp's standard function, `(random arg)`.

Input:

    upfrom, below. Each elements of returned tensor is in the range of: `[upfrom, below)`")
  
  (define-initializer-function
      ax+b
      (a b)
    (let ((a (coerce a (dtype->lisp-type (dtype tensor))))
	  (b (coerce b (dtype->lisp-type (dtype tensor)))))
      #'(lambda (i) (step-ax+b a i b)))
    "
The function ax+b is a family of initializer functions, and samples matrices from arithmetic progression.

```math
out_n = an + b
```

Inputs:

    a, b - Coefficients of the above formula."
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
    "

The function beta is a family of initializer functions, and sample matrices from beta distribution.

### Reference

1. Generating Beta Variates with Nonintegral Shape Parameters (R. C. H. Cheng University of Wales Institute of Science and Technology)


2. https://dl.acm.org/doi/pdf/10.1145/359460.359482


Note: My implementation is unstable, being occurs floating-overflow constantly..., especially when min(alpha, beta) < 1.0 (i.e.: beta-bc)")

  (define-initializer-function
      normal
      (mean stddev)
    (let* ((mean (coerce mean 'double-float))
	   (stddev (coerce stddev 'double-float)))
      #'(lambda (i)
	  (declare (ignore i))
	  (coerce (cl-randist:random-normal-ziggurat mean stddev) (dtype->lisp-type (dtype tensor)))))
    "The function normal is a family of initializer functions, and samples matrices from normal distribution.


### Reference

1. https://github.com/lvaruzza/cl-randist (seems to create ziggurat table with size=128)


### Inputs

mean

stddev - Standard Deviation, σ.
")

  (define-initializer-function
      randn
      ()
    (let* ((sampler (get-randn-sampler (dtype tensor))))
      #'(lambda (i)
	  (declare (ignore i))
	  (funcall sampler)))
    "The function randn is a family of initializer functions, and samples the gaussian distributions using ziggurat algorithm with table-size=256.


### References


1. https://andantesoft.hatenablog.com/entry/2023/04/30/183032

2. Marsaglia, G., & Tsang, W. W. (2000). The ziggurat method for generating random variables. Journal of statistical software.

3. https://marui.hatenablog.com/entry/2023/01/23/194507")

  (define-initializer-function
      exponential
      ()
    (let ((sampler (get-exponential-sampler (dtype tensor))))
      #'(lambda (i)
	  (declare (ignore i))
	  (funcall sampler)))
    "The function exponential is a family of initializer functions, and samples the exponential distribution using ziggurat algorithm with table-size=256.


### References

1. https://andantesoft.hatenablog.com/entry/2023/04/30/183032

2. Marsaglia, G., & Tsang, W. W. (2000). The ziggurat method for generating random variables. Journal of statistical software.

3. https://marui.hatenablog.com/entry/2023/01/23/194507")

  (define-initializer-function
      gamma
      (k)
    #'(lambda (i)
	(declare (ignore i))
	(sample-gamma (coerce k (dtype->lisp-type (dtype tensor)))))
    "The function gamma is a family of initializer functions, and samples matrices from the gamma distribution.

### References

1. https://github.com/lvaruzza/cl-randist
")

  (define-initializer-function
      bernoulli
      (p)
    #'(lambda (i)
	(declare (ignore i))
	(coerce (sample-bernoulli p) (dtype->lisp-type (dtype tensor))))
    "The bernoulli is a family of initializer functions, and samples matrices from bernoulli distribution.

### Inputs

p - Takes 1 with probability p and 0 with probalibity (1-p).")

  (define-initializer-function
      chisquare
      (df)
    #'(lambda (i)
	(declare (ignore i))
	(sample-gamma (coerce (/ df 2.0) (dtype->lisp-type (dtype tensor)))))
    "The function chisquare is a family of initializer functions, and samples matrices from chisquare distributions.

### Inputs

df - degree of freedom.

### References

 https://github.com/lvaruzza/cl-randist"))


(defmacro define-tensor-initializer (function-name
				     (&rest args)
				     initializer-lambda
				     document
				     &key (keep-order? nil))
  "```lisp
(define-tensor-initializer (function-name (&rest args) initializer-lambda document &key (keep-order? nil)))
```


`define-tensor-initializer` is a macro which is used to define a initializer function.


Initializer function is a function whose arguments follow this format:

    (function-name shape <Initializer's Arguments> &rest initargs &key &allow-other-keys)

Input:

    function-name - the function is defined after this argument

    args          - Initializer's Arguments

    initializer-lambda - A form to be expanded as the sampling function, which must return a function of #'(lambda (i) ...) where i is the index of element.

    keep-order? - set t if the index is needed to sampling matrices.

Example:

```lisp
(define-initializer-function
    uniform-random
    (upfrom below)
  (let ((upfrom (coerce upfrom (dtype->lisp-type (dtype tensor))))
	(below  (coerce below  (dtype->lisp-type (dtype tensor)))))
    #'(lambda (i)
	(declare (ignore i))
	(sample-uniform-random upfrom below)))
    \"\")

(uniform-random `(10 10) 0.1 0.3 :requires-grad t)

{CPUTENSOR[float] :shape (10 10)  
  ((0.13149574  0.15135926  0.1569588   ~ 0.103781514 0.20610212  0.19365484)                   
   (0.2638953   0.12672275  0.21630599  ~ 0.16542184  0.10228193  0.12928057)   
                ...
   (0.20429519  0.12252951  0.17538154  ~ 0.22072719  0.18642941  0.11027551)
   (0.14372297  0.11097031  0.25514898  ~ 0.28739202  0.18398522  0.15176433))
  :facet :exist
  :requires-grad T
  :backward NIL}
```

(Note that new tensor is binded to tensor, being used to determined dtype etc...)
"
  `(defun ,function-name (shape ,@args &rest initargs &key &allow-other-keys)
     ,document
     (let ((tensor (apply #'make-tensor shape initargs)))
       (initialize-vec! tensor ,initializer-lambda :keep-order? ,keep-order?)
       tensor)))


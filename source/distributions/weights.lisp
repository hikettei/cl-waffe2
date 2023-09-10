
(in-package :cl-waffe2/distributions)

;; Including compiling time ugh...

;; sampling initializer function

(defmacro define-tensor-initializer-export (function-name
					    (&rest args)
					    initializer-lambda
					    document
					    &key (keep-order? nil))
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (export ',function-name)
     (define-tensor-initializer ,function-name (,@args) ,initializer-lambda ,document :keep-order? ,keep-order?)))
  
(define-tensor-initializer-export
    xavier-uniform
    nil
    (let* ((in-features  (car    (last shape 2)))
	   (out-features (second (last shape 2)))
	   (coeff (sqrt (/ 6 (+ in-features out-features)))))
      #'(lambda (i)
	  (declare (ignore i))
	  (* coeff (sample-uniform-random -1.0 1.0))))
    "")

(define-tensor-initializer-export
    xavier-gaussian
    nil
    (let* ((in-features  (car    (last shape 2)))
	   (out-features (second (last shape 2)))
	   (stddev (coerce (sqrt (/ 2 (+ in-features out-features))) 'double-float)))
      #'(lambda (i)
	  (declare (ignore i))
	  (coerce (cl-randist:random-normal-ziggurat 0.0d0 stddev) (dtype->lisp-type (dtype tensor)))))
    "")

;; He/Orthogonal


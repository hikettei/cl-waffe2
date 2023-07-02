
(in-package :cl-waffe2/distributions)

;; Including compiling time ugh...

;; sampling initializer function


(define-tensor-initializer
    xavier-uniform
    nil
    (let* ((in-features  (car    (last shape 2)))
	   (out-features (second (last shape 2)))
	   (coeff (sqrt (/ 6 (+ in-features out-features)))))
      #'(lambda (i)
	  (declare (ignore i))
	  (* coeff (sample-uniform-random -1.0 1.0))))
    "")

(define-tensor-initializer
    xavier-gaussian
    nil
    (let* ((in-features  (car    (last shape 2)))
	   (out-features (second (last shape 2)))
	   (stddev (sqrt (/ 2 (+ in-features out-features))))
	   (sampler (get-randn-sampler (dtype tensor))))
      #'(lambda (i)
	  (declare (ignore i))
	  (* stddev (funcall sampler))))
    "")


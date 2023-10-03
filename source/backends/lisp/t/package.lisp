
(in-package :cl-user)

(defpackage :cl-waffe2/backends.lisp.test
  (:use :cl
        :fiveam
        :cl-waffe2
        :cl-waffe2/backends.lisp
        :cl-waffe2/distributions
        :cl-waffe2/base-impl
        :cl-waffe2/base-impl.test
        :cl-waffe2/vm.generic-tensor
        :cl-waffe2/vm.nodes))

(in-package :cl-waffe2/backends.lisp.test)

(def-suite :lisp-backend-test)

(in-suite  :lisp-backend-test)

;;(eval-when (:compile-toplevel :load-toplevel :execute)
  
(add-tester LispTensor)
(sub-tester LispTensor)
(mul-tester LispTensor)
(div-tester LispTensor)
(move-tester LispTensor)

(scalar-add-tester LispTensor)
(scalar-sub-tester LispTensor)
(scalar-mul-tester LispTensor)
(scalar-div-tester LispTensor)

(sum-tester LispTensor)

(mathematical-test-set LispTensor)

(max-tester LispTensor)
(min-tester LispTensor)

(comparison-test-set LispTensor)
;;)

(defun ~= (x y)
  (- (abs (- x y)) 0.00001))

(defun lazy-test ()
  (let ((result1
	  (with-devices (LispTensor)
	    (tensor-vec (proceed (lazy #'sin (!permute (ax+b `(10 10 10) 1 0) 2 0 1))))))
	(result2
	  (tensor-vec (proceed (!sin (!permute (ax+b `(10 10 10) 1 0) 2 0 1))))))
    (every #'~= result1 result2)))


(defun lazy-diff-p ()
  (let* ((a1 (parameter (ax+b `(3 3) 1 0)))
	 (a2 (parameter (ax+b `(3 3) 1 0))))

    (proceed-backward
     (lazy #'sin a1 :diff #'cos))

    (proceed-backward
     (!sin a2))

    (every #'~= (tensor-vec (grad a1)) (tensor-vec (grad a2)))))

(defun lazy-reduce-test ()
  (let ((result1
	  (with-devices (LispTensor)
	    (tensor-vec (proceed (lazy-reduce #'max (!permute (ax+b `(10 10 10) 1 0) 2 0 1))))))
	(result2
	  (tensor-vec (proceed (!max (!permute (ax+b `(10 10 10) 1 0) 2 0 1) :axis 1)))))
    (every #'~= result1 result2)))

(test lazy-op-test
  (is (lazy-test)))

(test lazy-diff-p
  (is (lazy-diff-p)))

(test lazy-reduce-test-max
  (is (lazy-reduce-test)))



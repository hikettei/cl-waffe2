
(in-package :cl-waffe2/backends.cpu.test)

(in-suite :test-backends-cpu)

(eval-when (:compile-toplevel :load-toplevel :execute)
  
(add-tester CPUTensor)
(sub-tester CPUTensor)
(move-tester CPUTensor)

(matmul-test-set CPUTensor)

)

(defun max-diff-test ()
  (let ((a (parameter (cl-waffe2/distributions:ax+b `(16 16) 1 0))))
    (proceed-backward
     (!max a))

    (= 16.0
       (tensor-vec
	(proceed
	 (->scal
	  (!sum
	   (grad a))))))))

(defun min-diff-test ()
  (let ((a (parameter (cl-waffe2/distributions:ax+b `(16 16) 1 0))))
    (proceed-backward
     (!min a))

    (= 16.0
       (tensor-vec
	(proceed
	 (->scal
	  (!sum
	   (grad a))))))))

(test maxmin-diff-test
  (is (max-diff-test))
  (is (min-diff-test)))

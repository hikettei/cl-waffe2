
(in-package :cl-waffe2/backends.cpu.test)

(in-suite :test-backends-cpu)

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

(defun matmul-form-1 ()
  (proceed (!matmul (ax+b `(10 10) 1 0) (ax+b `(10 10) 1 0))))

(defun matmul-form-2 ()
  (proceed (!matmul (ax+b `(10 3) 1 0) (ax+b `(3 10) 1 0))))

(defun matmul-form-3 ()
  (proceed (!matmul (!t (ax+b `(3 10) 1 0)) (ax+b `(3 10) 1 0))))

(defun matmul-form-4 ()
  (proceed (!matmul (!t (ax+b `(3 10) 1 0)) (!t (ax+b `(10 3) 1 0)))))

(defun matmul-form-5 ()
  (cl-waffe2::with-dtype :double
    (proceed (!matmul (ax+b `(10 10) 1 0) (ax+b `(10 10) 1 0)))))

(defun !tc (x)
  (->contiguous (!t x)))

(test row-major-mm-test
  (is (every #'=
	     (tensor-vec (matmul-form-1))
	     (tensor-vec (proceed (!tc (cl-waffe2::with-row-major (matmul-form-1)))))))
  (is (every #'=
	     (tensor-vec (matmul-form-2))
	     (tensor-vec (proceed (!tc (cl-waffe2::with-row-major (matmul-form-2)))))))
  (is (every #'=
	     (tensor-vec (matmul-form-3))
	     (tensor-vec (proceed (!tc (cl-waffe2::with-row-major (matmul-form-3)))))))
  (is (every #'=
	     (tensor-vec (matmul-form-4))
	     (tensor-vec (proceed (!tc (cl-waffe2::with-row-major (matmul-form-4)))))))
  (is (every #'=
	     (tensor-vec (matmul-form-5))
	     (tensor-vec (proceed (!tc (cl-waffe2::with-row-major (matmul-form-5))))))))


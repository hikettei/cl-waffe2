
(in-package :cl-waffe2/backends.jit.cpu.test)

(in-suite :jit-cpu-test)

;; TODO: Write tests

;; A+=B (!copy (!view ...))
;; CopyとView以外は動く -> それさえ動けばSoftmax ugoku
;; with-no-grad

;;(test arithmetic-[normal+normal])

(defun M= (a b)
  (every #'= a b))

(defmacro with-test-form (&body body)
  `(flet ((lisp-case ()
	    (with-devices (cl-waffe2/backends.lisp:LispTensor)
	      ,@body))
	  (jit-case ()
	    (with-cpu-jit ()
	      ,@body)))
     (M=
      (tensor-vec (proceed (lisp-case)))
      (tensor-vec (proceed (jit-case))))))


(test arithmetic-A+B
  (is (with-cpu-jit ()
	(M=
	 (tensor-vec
	  (proceed
	   (!add (ax+b `(3 3) 1 0) (ax+b `(3 3) 1 0))))
	 (tensor-vec
	  (ax+b `(3 3) 2 0)))))
  (is (with-cpu-jit ()
	(M=
	 (tensor-vec
	  (proceed
	   (!sub (ax+b `(3 3) 1 0) (ax+b `(3 3) 1 0))))
	 (tensor-vec
	  (ax+b `(3 3) 0 0))))))

(test arithmetic-[broadcast]A+B
  (is (with-cpu-jit ()
	(M=
	 (tensor-vec
	  (proceed
	   (!add
	    (!view (ax+b `(1 3) 0 0) `(:broadcast 3))
	    (ax+b `(3 3) 0 1))))
	 (tensor-vec
	  (ax+b `(1 3) 0 3)))))
  (is (with-test-form
	(!sum (ax+b `(3 3) 1 0)))))


(test softmax-complicated-views-test
  (is (with-test-form (cl-waffe2/nn:!softmax (ax+b `(3 3) 0 1)))))




(in-package :cl-waffe2/backends.jit.cpu.test)

(in-suite :jit-cpu-test)


;; Testing call-with-view/In-place mutation are working well
;; Also: do tests on with-no-grad mode.

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

(test softmax-complicated-views-test
  (is (with-test-form (cl-waffe2/nn:!softmax (ax+b `(3 3) 0 1)))))


;; Things should also tested:
;; backward

(test backward-test
  (is (let ((a (parameter (ax+b `(3 3) 0 1))))
	(proceed-backward (!sin a))
	(= (cos 1) (vref (grad a) 0)))))

;;(progn
;;  (let ((a (parameter (ax+b `(3 3) 0 1))))	      
 ;;   (forward (build (!mul a (make-tensor 1.0))))))


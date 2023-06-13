
(in-package :cl-waffe2/base-impl.test)

(in-suite :base-impl-test)

(define-tester sum-tester :dense
  (let ((a (make-tensor `(100 100 100) :initial-element 1.0)))
    (when
	(and (= (vref (proceed (!sum a)) 0) (* 100 100 100))
	     (= (vref (proceed (!sum a :axis 0)) 0) 100)
             (= (vref (proceed (!sum a :axis 1)) 0) 100)
	     (= (vref (proceed (!sum a :axis -1)) 0) 100)
	     (= (vref (proceed (!sum a :axis `(0 1))) 0) (* 100 100)))
      ;; forward -> passed
      (let ((k (make-tensor `(100 100) :initial-element 1.0 :requires-grad t))
	    (m (make-tensor `(100 100) :initial-element 1.0 :requires-grad t)))
	(proceed-backward (!sum k))
	(proceed-backward (!sum m :axis 0))
	(if (and
	     (= (vref (grad k) 0) (/ 1 (* 100 100)))
	     (= (vref (grad m) 0) (/ 1 100)))
	    t
	    :backward)))))

(sum-tester cl-waffe2/backends.lisp:LispTensor)

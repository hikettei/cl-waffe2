
(in-package :cl-waffe2/vm.nodes.test)

(in-suite :test-nodes)


(defmodel (SinModel1 (self)
	   :where (A[~] -> B[~])
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (!sin x))))

(defmodel (CosModel1 (self)
	   :where (A[~] -> B[~])
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (!cos x))))

(defmodel (MulModel1 (self)
	   :where (A[~] B[~] -> C[~])
	   :on-call-> ((self x y)
		       (declare (ignore self))
		       (!mul x y))))

(defmodel-as (SinModel1) :named !sin-static)
(defmodel-as (CosModel1) :named !cos-static)
(defmodel-as (MulModel1) :named !mul-static)


(define-op (SinNode-Static (self)
	    :where (A[~] -> A[~])

	    :save-for-backward-names (x)
	    :forward ((self x)
		      (with-setting-save4bw ((x x))
			(!sin-static x)))
	    :backward ((self dout)
		       (with-reading-save4bw ((x x))
			 (!mul-static dout (!cos-static x))))))

(defun test-composite-diff ()
  (let ((a (parameter (ax+b `(3 3) 0 1))))
    (proceed-backward (call (SinNode-Static) a))
    (~= (cos 1) (vref (grad a) 0))))

(defun composite-with-build ()
  (let* ((a (parameter (ax+b `(3 3) 0 1)))
	 (model (build (call (SinNode-Static) a))))
    (forward model)
    (backward model)
    (let ((f1 (~= (cos 1) (vref (grad a) 0))))
      (forward model)
      (backward model)

      ;; Gradients are resetted...
      (let ((f2 (~= (cos 1) (vref (grad a) 0))))
	(forward model)
	(backward model)

	(and f1 f2
	     (~= (+ (cos 1))
		 (vref (grad a) 0)))))))

(defun composite-with-build1 ()
  (let* ((a (parameter (ax+b `(3 3) 0 1)))
	 (model (build (!sum (call (SinNode-Static) (!sin a)))))
	 (grad (* (cos (sin 1)) (cos 1))))
    (forward model)
    (backward model)
    (let ((f1 (~= (vref (grad a) 0) grad)))
      (forward model)
      (backward model)
      (and f1
	   (~= (vref (grad a) 0) (+ grad grad))))))

(test composite-static-function-diff-test
  (is (test-composite-diff))
  (is (composite-with-build))
  (is (composite-with-build1)))


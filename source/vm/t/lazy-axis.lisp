
(in-package :cl-waffe2/vm.test)

(in-suite :vm-test)

(defun lazy-axis-net-1 ()
  (let ((out (build
	      (!add
	       (make-input `(A B) :X)
	       (make-tensor
		(make-lazyaxis `(+ A B))))
	      :inputs `(:X))))
    (progn
      (= 7.0 (vref (forward out (ax+b `(3 4) 0 0)) 0)))))

(defun lazy-axis-net-adjust-later ()
  (let ((out (build
	      (!add
	       (make-input `(A B) :X)
	       (make-tensor
		(make-lazyaxis `(+ A B))))
	      :inputs `(:X))))
    (and
     (= 7.0 (vref (forward out (ax+b `(3 4) 0 0)) 0))
     (= 8.0 (vref (forward out (ax+b `(4 4) 0 0)) 0)))))

(test lazy-axis-scalar-test
  (is (lazy-axis-net-1))
  (is (lazy-axis-net-adjust-later)))

;; ReshapeTest

(!reshape (randn `(3 3 3 3)) (~ N C H W -> N C H W))

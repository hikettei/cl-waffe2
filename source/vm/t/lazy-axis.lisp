
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

;; ScalarTensors with LazyAxis works well?
(test lazy-axis-scalar-test
  (is (lazy-axis-net-1))
  (is (lazy-axis-net-adjust-later)))

(defun conv2d-forward-test ()
  (call (Conv2D 3 6 `(5 5)) (make-input `(N 3 25 25) nil)))

(defun avg-pool2d-forward-test ()
  (call (AvgPool2d `(5 5)) (make-input `(N 3 25 25) nil)))

(defun max-pool2d-forward-test ()
  (call (MaxPool2d `(5 5)) (make-input `(N 3 25 25) nil)))


;; Testing just a node construction
(test dynamic-cnn-node-construction-test
  (is (conv2d-forward-test))
  (is (avg-pool2d-forward-test))
  (is (max-pool2d-forward-test)))

(defsequence LazyCNN (&key
		      (out-channels1 4)
		      (out-channels2 16))
	     (Conv2D 1 out-channels1 `(3 3))
	     (asnode #'!relu)     
	     (MaxPool2D    `(2 2))
	     (Conv2D out-channels1 out-channels2 `(5 5))
	     (asnode #'!relu)
	     (MaxPool2D `(2 2))
	     (asnode #'!reshape t (* 16 4 4)) 
	     (LinearLayer (* 16 4 4) 10))

(defun cnn-build-test ()
  (build (call (LazyCNN) (make-input `(N 1 28 28) :X)))
  )
;; ReshapeTest
;;(print (!reshape (make-input `(3 3 3 3)) (~ N C H W -> N C H W)))

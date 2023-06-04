
(in-package :cl-waffe2/vm.nodes.test)

(in-suite :test-nodes)

(defun test-subscript (subscript input out)
  (let ((f (cl-waffe2/vm.nodes::create-subscript-p subscript)))
    (multiple-value-bind (res errors) (funcall f input)
      (and (null errors)
	   (equal res out)))))

(defun error-test-subscript (subscript input)
  (let ((f (cl-waffe2/vm.nodes::create-subscript-p subscript)))
    (multiple-value-bind (res errors) (funcall f input)
      (declare (ignore res))
      (and errors))))

(test test-simple-subscript-p
  (is (test-subscript
       `([x] -> [x])
       `((10))
       `((10))))
  (is (test-subscript
       `([x y] -> [x y])
       `((10 10))
       `((10 10))))
  (is (test-subscript
       `([x y] -> [x y])
       `((10 a))
       `((10 a))))
  (is (test-subscript
       `([x y] -> [y x])
       `((10 5))
       `((5 10))))
  (is (test-subscript
       `([x y z] -> [x y z])
       `((10 10 10))
       `((10 10 10))))
  (is (test-subscript
       `([x y z l] -> [x y z l])
       `((10 10 10 10))
       `((10 10 10 10))))
  (is (test-subscript
       `([x y z l] -> [x y z])
       `((5 3 2 10))
       `((5 3 2)))))


(test test-adjustable-input-subscript-p
  ;; It is controversial whether we allow it or not.
  (is (test-subscript
       `([~ x] -> [x])
       `((10))
       `((10))))
  (is (test-subscript
       `([~ x] -> [x])
       `((10 3))
       `((3))))
  (is (test-subscript
       `([~ x y] -> [x y])
       `((5 3 2))
       `((3 2))))
  (is (test-subscript
       `([~ x y] -> [y x])
       `((2 3 4 10 5))
       `((5 10))))
  (is (test-subscript
       `([~ x y z] -> [x y z])
       `((100 10 2 10))
       `((10 2 10))))
  (is (test-subscript
       `([~ x y z] -> [z y x])
       `((100 10 2 10))
       `((10 2 10))))
  (is (test-subscript
       `([~ x y z l] -> [x])
       `((10 10 10 10))
       `((10)))))


(test test-adjustable-out-subscript-p
  (is (test-subscript
       `([~ x] -> [~ x])
       `((10 3))
       `((10 3))))
  (is (test-subscript
       `([~ x y] -> [~ x y])
       `((5 3 2))
       `((5 3 2))))
  (is (test-subscript
       `([~ x y] -> [~ y x])
       `((2 3 4 10 5))
       `((2 3 4 5 10))))
  (is (test-subscript
       `([~ x y z] -> [~ x y z])
       `((100 10 2 10))
       `((100 10 2 10))))
  (is (test-subscript
       `([~ x y z] -> [~ z y x])
       `((100 10 2 10))
       `((100 10 2 10))))
  (is (test-subscript
       `([~ x y z l] -> [~ x])
       `((10 10 5 3 2))
       `((10 10)))))

(test test-xy-subscript-p
  (is (test-subscript
       `([x y] [x y] -> [x y])
       `((10 a) (10 3))
       `((10 3))))
  (is (test-subscript
       `([x y] [x y] [x y] -> [x y])
       `((10 a) (10 3) (10 3))
       `((10 3))))
  (is (test-subscript
       `([~ x] [~ x] -> [~ x])
       `((10 3) (10 3))
       `((10 3))))
  (is (test-subscript
       `([~ x y] [~ x y]-> [~ x y])
       `((5 3 2) (5 3 2))
       `((5 3 2))))
  (is (test-subscript
       `([~ x y] [~ x y]-> [~ y x])
       `((5 3 2) (5 3 2))
       `((5 2 3))))
  (is (test-subscript
       `([~ x y] [~ z]-> [x z])
       `((10 3 2) (10 3 2))
       `((3 2)))))

(test test-xyz-subscript-p
  (is (test-subscript
       `([~ x] [~ x] [~ x] [~ x] -> [~ x])
       `((10 3) (10 3) (10 3) (10 3))
       `((10 3))))
  (is (test-subscript
       `([~ x] [~ y] [~ x] [~ y] -> [x y])
       `((10 3) (10 5) (10 3) (10 5))
       `((3 5)))))

(test test-complicated-subscript-p
  (is (test-subscript
       `([~ x] -> [~ x] [x])
       `((10 3))
       `((10 3) (3))))
  (is (test-subscript
       `([~ x] -> [~ x] [x] [x])
       `((10 3))
       `((10 3) (3) (3))))
  (is (test-subscript
       `([x y] -> [x y] [y x] [x y])
       `((10 3))
       `((10 3) (3 10) (10 3)))))

(test test-where-phase
  (is (test-subscript
       `([~ x] -> [~ x] [y] where y = 1)
       `((10 3))
       `((10 3) (1))))
  (is (test-subscript
       `([x y] -> [k l] where k = 20 l = 6)
       `((10 3))
       `((20 6))))
  (is (test-subscript
       `([x y] -> [out] where out = `(10 10 10))
       `((10 3))
       `((10 10 10))))
  (is (test-subscript
       `([x y] -> [x out] where out = `(10 10 10))
       `((10 3))
       `((10 10 10 10))))
  ;;(is (test-subscript
  ;;     `([a] [b] -> [b] where a = `(10 10 10) b = `(2 2 2))
  ;;     `((10 10 10) (2 2 2))
  ;;     `((2 2 2))))

  )

(test test-error-subscript
  (is (error-test-subscript
       `([~ x y] [~ y] -> [~ y])
       `((10 5 2) (3 2))))

  ;; BUG. ~ = nil is not allowed.
  (is (error-test-subscript
       `([~ x y] [~ y] -> [~ y])
       `((10 5 2) (2))))
  
  (is (error-test-subscript
       `([~ x y] [x y z] -> [~ y])
       `((10 5 2) (3 2))))

  (is (error-test-subscript
       `([~ x y z] [x y z] -> [~ y])
       `((10 5 2) (3 2 3))))
  
  (is (error-test-subscript
       `([~] [~] -> [~])
       `((10 5 2) (3 2 3))))

  (is (error-test-subscript
       `([~] [~] -> [~])
       `((10 a) (10 b))))

  (is (error-test-subscript
       `([x y] [x y] [x y] -> [x y])
       `((10 a) (10 1) (10 2))))
  
  (is (error-test-subscript
       `([x y] [x y] [x y] -> [x y])
       `((10 a) (10 b) (10 b))))
  
  (is (error-test-subscript
       `([x y] [x y]-> [x y])
       `((10 a) (10 b))))

  ;; Should be added more... (After deciding specifications)

  )

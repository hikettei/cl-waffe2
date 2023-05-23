
(in-package :cl-waffe2/vm.nodes.test)

(in-suite :test-nodes)

(defun test-subscript (subscript input out)
  (let ((f (cl-waffe2/vm.nodes::create-subscript-p subscript)))
    (multiple-value-bind (res errors) (funcall f input)
      (and (null errors)
	   (equal res out)))))

(defun error-test-subscript (subscript input out)
  (let ((f (cl-waffe2/vm.nodes::create-subscript-p subscript)))
    (multiple-value-bind (res errors) (funcall f input)
      (and errors))))

;; error, with-let, out~, multiple-inputs/outputs
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
       `([~ x y z] -> [z y x])
       `((100 10 2 10))
       `((10 2 10))))
  (is (test-subscript
       `([~ x y z l] -> [x])
       `((10 10 10 10))
       `((10)))))

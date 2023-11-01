
(in-package :cl-waffe2/vm.iterator)

(defpackage :cl-waffe2/vm.iterator.test
  (:use
   :cl
   :fiveam
   :cl-waffe2/vm.iterator))

(in-package :cl-waffe2/vm.iterator.test)

(def-suite :iterator-test)
(in-suite  :iterator-test)

;; Expected to be working as well as for dynamic shapes
(test range-size-computation
  (is (= 10 (range-size
	     (range 0 10 1))))
  (is (= 5  (range-size
	     (range 0 10 2))))
  (is (= 5  (range-size
	     (range 0 10 -2)))))


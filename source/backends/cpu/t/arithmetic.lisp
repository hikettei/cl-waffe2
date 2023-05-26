
(in-package :cl-waffe2/backends.cpu.test)

(in-suite :test-backends-cpu)


(test test-arithmetic
  (is (forward (AddNode nil) (make-tensor `(10 10)) (make-tensor `(10 10)))))


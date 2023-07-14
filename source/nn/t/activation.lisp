
(in-package :cl-waffe2/nn.test)

(in-suite :nn-test)

(defun relu-test ()
  (let ((a (randn `(10 10))))
    (let ((out (proceed (!relu a))))
      (every #'(lambda (x) (>= x 0)) (tensor-vec out)))))

(defun softmax-test ()
  (let ((a (randn `(10 10))))
    (let ((out (proceed (->scal (!sum (!softmax a))))))
      (< (abs (- 10.0 (tensor-vec out))) 0.0001))))


(test activation-test
  (is (relu-test))
  (is (softmax-test)))


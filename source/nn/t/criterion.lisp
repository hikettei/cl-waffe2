
(in-package :cl-waffe2/nn.test)

(in-suite :nn-test)

(defun softmax-cross-entropy-test ()
  (let ((a (parameter (randn `(10 10))))
	(b (parameter (randn `(10 10)))))
    (proceed (!sum (softmax-cross-entropy a b)))))



;; Tests for define-static-node's backward.
(defun softmax-cross-entropy-test1 ()
  (let ((a (parameter (randn `(10 10))))
	(b (parameter (randn `(10 10)))))

    (proceed-backward (!sum (softmax-cross-entropy (!relu a) (!relu b))))
    (and
     (some #'(lambda (x) (not (= x 0))) (tensor-vec (grad a)))
     (some #'(lambda (x) (not (= x 0))) (tensor-vec (grad b))))))
    

(test softmax-cross-entropy-and-static-node-backward
  (is (softmax-cross-entropy-test))
  (is (softmax-cross-entropy-test1)))




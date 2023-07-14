
(in-package :cl-waffe2/nn.test)

(in-suite :nn-test)

(defun softmax-cross-entropy-test ()
  (let ((a (parameter (randn `(10 10))))
	(b (parameter (randn `(10 10)))))
    (proceed (!sum (softmax-cross-entropy a b)))))



(defun softmax-cross-entropy-test1 ()
  (let ((a (parameter (randn `(10 10))))
	(b (parameter (randn `(10 10)))))
    ;; Tests differentiable composite node
    ;; And... Maybe it is now workig :(
    (proceed-backward (!sum (!add (!relu a) (!relu b))))
    (values (grad a) (grad b))))
    

(test softmax-cross-entropy
  (is (softmax-cross-entropy-test))
  (is (softmax-cross-entropy-test1)))



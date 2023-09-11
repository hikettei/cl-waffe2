
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
    ;;(reset-compiled-function-cache!)
    (proceed-backward (!sum (softmax-cross-entropy (!relu a) (!relu b))))
    (some #'(lambda (x) (not (= x 0))) (tensor-vec (grad a)))))

;; BugFix

;; (reset-compiled-function-cache!)
;; (softmax-cross-entropy-test)
;; (softmax-cross-entropy-test1)
;; -> error due to MoveTensorNode was ignored.
;; -> self was duplicated in the cached functions.

(test softmax-cross-entropy-and-static-node-backward
  (is (softmax-cross-entropy-test))
  (is (softmax-cross-entropy-test1)))

(defun criterion-with-build ()
  (let ((a (parameter (randn `(10 10))))
	(b (parameter (randn `(10 10)))))
    (let ((model (build (!sum (softmax-cross-entropy (!relu a) (!relu b)))
			:compile-mode :safety)))
      (forward model)
      (backward model)

      ;;zero-grads! model)
      (print (grad a))
      (print (grad b))

      (forward model)
      (backward model)

      (print (Grad a))
      (print (grad b))
      )))

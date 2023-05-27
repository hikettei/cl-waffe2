

(in-package :cl-waffe2/vm.nodes.generic-tensor)

(in-suite :test-tensor)


;; Problems: Compile-Speed
;; (10 a) (10 10) <- a = 10
(defun test1 ()
  (let* ((input (make-input `(10 10) :train-x))
	 (out (construct-forward (forward (AddNode) input (make-tensor `(10 10))))))
    (embody-input input (make-tensor `(10 10)))
    (print (time (funcall out)))))

(test construct-tensor
  (is (test1)))



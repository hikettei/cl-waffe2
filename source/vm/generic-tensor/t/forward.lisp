

(in-package :cl-waffe2/vm.nodes.generic-tensor)

(in-suite :test-tensor)


;; Problems: Compile-Speed (add config) (there's no need)
;; (10 a) (10 10) <- a = 10 (DONE)

;; Making Add/Sub/Mul/Div Nodes
;; OpenBLAS/Common Lisp backend.
;; update: construct-forward (InputTensor UI)
;; add:    construct-backward
;; view -> view.
;; ViewNode

(defun test-simple-forward ()
  (with-single-device (LispTensor)
    (let ((out (!add (make-tensor `(10 10))
		     (make-tensor `(10 10)))))
      (funcall (construct-forward out)))))

(test test-forward
  (is (test-simple-forward)))

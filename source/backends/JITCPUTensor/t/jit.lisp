
(in-package :cl-waffe2/backends.jit.cpu.test)

(in-suite :jit-cpu-test)

(defun complicated-index ()
  (with-cpu-jit (CPUTensor)
    (let ((out
	    (!add
	     (!view (make-input `(N C H W) :X) `(H W))
	     (!view (make-input `(N C H W) :Y) `(H W)))))
      (build out :inputs `(:X :Y)))))

(test complicated-index-test
  (is (forward
       (complicated-index)
       (randn `(100 20 30 40))
       (randn `(100 20 30 40)))))



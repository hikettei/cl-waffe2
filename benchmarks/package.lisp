
(in-package :cl-user)

(defpackage :cl-waffe2/benchmark
  (:use :cl
        :cl-waffe2
	:cl-waffe2/vm.generic-tensor
        :cl-waffe2/vm.nodes
        :cl-waffe2/base-impl
   :cl-waffe2/distributions))

(in-package :cl-waffe2/benchmark)

(defun mm-bench (n &key (times 1000))
  (with-no-grad
    (let ((model (build (!matmul (randn `(,n ,n))
				 (randn `(,n ,n)))
			:compile-mode :fastest)))
      (let ((t1 (get-internal-real-time)))
	(dotimes (i times)
          (forward model))
	(let ((t2 (get-internal-real-time)))
	  (float (/ (- t2 t1) internal-time-units-per-second)))))))


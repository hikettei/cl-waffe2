
(in-package :cl-user)

(defpackage :cl-waffe2/distributions
  (:use :cl :cl-waffe2/vm.generic-tensor))

(in-package :cl-waffe2/distributions)

;; If num-cores = 1 work with pure Lisp
;; If num-cores = 2, 3, 4.... works with lparallel pfuncall

;; TODO: lparallel, Fix row-major, matmul. SVD
;; TODO: Test Cases, Plotting randn


(in-package :cl-user)

(defpackage :cl-waffe2/distributions
  (:use :cl :cl-waffe2/vm.generic-tensor :cl-waffe2/threads :cl-waffe2/base-impl)
  (:export #:define-tensor-initializer))

(in-package :cl-waffe2/distributions)



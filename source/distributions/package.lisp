
(in-package :cl-user)

(defpackage :cl-waffe2/distributions
  (:nicknames #:wf/d)
  (:use :cl :cl-waffe2/vm.generic-tensor :cl-waffe2/threads :cl-waffe2/base-impl)
  (:export #:define-tensor-initializer #:define-with-typevar :define-with-typevar-dense #:define-with-typevar-sparse))

(in-package :cl-waffe2/distributions)




(in-package :cl-user)

(defpackage :cl-waffe2/backends.jit.lisp.test
  (:use :cl :cl-waffe2 :cl-waffe2/distributions :cl-waffe2/base-impl :cl-waffe2/vm.generic-tensor :cl-waffe2/vm.nodes :cl-waffe2/backends.lisp :cl-waffe2/backends.jit.lisp :fiveam))

(in-package :cl-waffe2/backends.jit.lisp.test)

(def-suite :jit-lisp-test)


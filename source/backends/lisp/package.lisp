
(in-package :cl-user)

(defpackage :cl-waffe2/backends.lisp
  (:documentation "cl-waffe2/backends.lisp provides LispBackend provides the Lisp-Backend, which is portable. (only using ANSI Common Lisp)")
  (:use :cl :cl-waffe2/vm.generic-tensor
	:cl-waffe2/vm.nodes
        :cl-waffe2/base-impl)
  (:export
   #:LispTensor))

(in-package :cl-waffe2/backends.lisp)

#-sbcl(setf cl-waffe2/vm.generic-tensor:*using-backend* `(cl-waffe2/backends.lisp:LispTensor))


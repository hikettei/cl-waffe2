
(in-package :cl-user)

(defpackage :cl-waffe2/backends.jit.lisp
  (:documentation "cl-waffe2/backends.jit.lisp demonstrates an example case of implementing jit with cl-waffe2.")
  (:use :cl
	:cl-waffe2/distributions
        :cl-waffe2/vm.generic-tensor
	:cl-waffe2/vm.nodes
        :cl-waffe2/base-impl)
  (:export
   #:JITLispTensor))


(in-package :cl-waffe2/backends.jit.lisp)



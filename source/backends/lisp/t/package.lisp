
(in-package :cl-user)

(defpackage :cl-waffe2/backends.lisp.test
  (:use :cl
        :fiveam
        :cl-waffe2
        :cl-waffe2/base-impl
        :cl-waffe2/base-impl.test
        :cl-waffe2/vm.generic-tensor
        :cl-waffe2/vm.nodes))

(in-package :cl-waffe2/backends.lisp.test)

(def-suite :lisp-backend-test)
(in-suite  :lisp-backend-test)

(add-tester LispTensor)
(sub-tester LispTensor)
(mul-tester LispTensor)
(div-tester LispTensor)

(scalar-add-tester LispTensor)
(scalar-sub-tester LispTensor)
(scalar-mul-tester LispTensor)
(scalar-div-tester LispTensor)

(sum-tester LispTensor)

(mathematical-test-set LispTensor)


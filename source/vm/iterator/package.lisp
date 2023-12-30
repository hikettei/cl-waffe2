
(in-package :cl-user)

(defpackage :cl-waffe2/vm.iterator
  (:nicknames #:wf/iter)
  (:use :cl :cl-waffe2/vm :cl-waffe2/vm.generic-tensor :alexandria)
  (:export
   #:range
   #:range-size
   #:range-from
   #:range-start-index
   #:range-to
   #:range-step
   #:range-nth
   #:do-range
   #:.range)
  (:export
   #:trace-invocation
   ))

(in-package :cl-waffe2/vm.iterator)

(defun butnil (list) (loop for l in list if l collect l))

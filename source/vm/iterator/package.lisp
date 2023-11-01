
(in-package :cl-user)

(defpackage :cl-waffe2/vm.iterator
  (:nicknames #:wf/iter)
  (:use :cl :cl-waffe2/vm)
  (:export
   #:range
   #:range-size
   #:range-from
   #:range-to
   #:range-step
   #:do-range
   #:.range))

(in-package :cl-waffe2/vm.iterator)




(in-package :cl-user)

;; Abstract Testing Tool

(defpackage :cl-waffe2/tester
  (:use :cl :rove)
  (:export
   #:running-test
   ))

(in-package :cl-waffe2/tester)

(defun running-test (&key (style :dot))
  (cl-waffe2:show-backends)
  (rove:run :cl-waffe2/test :style style))



(in-package :cl-user)

;; Abstract Testing Tool

(defpackage :cl-waffe2/tester
  (:use :cl :rove)
  (:export
   #:running-test
   ))

(in-package :cl-waffe2/tester)

;; [TODO] Roveで全部書き直す
;; [TODO] unittest
;; [TODO] Colorred timestamp
(defun running-test (&key (style :dot))
  ;; timestamp
  (cl-waffe2:show-backends)

  ;; Test all ops defined in base-impl
  (rove:run :cl-waffe2/test :style style))






(in-package :cl-user)

(defpackage :cl-waffe2/threads
  (:use :cl :lparallel)
  (:export
   #:*num-cores*
   #:with-num-cores
   #:multithread-p))

(in-package :cl-waffe2/threads)

;; MultiThreading configs

(defparameter *num-cores* 1 "If 1, Ignored.")

(defmacro with-num-cores ((num-core) &body body)
  `(let ((*num-cores* ,num-core))
     ,@body))

(defun multithread-p ()
  (not (= *num-cores* 1)))



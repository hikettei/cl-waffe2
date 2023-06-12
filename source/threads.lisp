
(in-package :cl-user)

(defpackage :cl-waffe2/threads
  (:use :cl :lparallel)
  (:export
   #:*num-cores*
   #:with-num-cores
   #:multithread-p
   #:maybe-with-lparallel
   #:maybe-pfuncall))

(in-package :cl-waffe2/threads)

;; MultiThreading configs

(defparameter *num-cores* 1 "If 1, Ignored.")

(defmacro with-num-cores ((num-core) &body body)
  `(let ((*num-cores* ,num-core))
     ,@body))

(defun multithread-p ()
  (not (<= (the fixnum *num-cores*) 1)))

(defmacro maybe-with-lparallel (&body body)
  `(let ((*kernel* (or *kernel* (if (multithread-p)
				    (make-kernel *num-cores*)
				    nil))))
     ,@body))

(defmacro maybe-pfuncall (function &rest args)
  `(if (multithread-p)
       (pfuncall ,function ,@args)
       (funcall ,function ,@args)))


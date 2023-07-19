
(in-package :cl-waffe2/backends.jit.lisp)

;; = [An blueprint of user-defined JIT Compiler in cl-waffe2] ===================
;;
;; 1. Goal
;;
;; Ex> (!sin (!sin (!sin x))) generates:
;;
;; (loop-with-view ...
;;          (setf (aref out ...) (sin (sin (sin i))))
;;
;; Without JIT:
;;
;; (setq out1 (loop-with-view (setf (aref out ...) (sin x))))
;; (setq out1 (loop-with-view (setf (aref out ...) (sin x))))
;; (setq out1 (loop-with-view (setf (aref out ...) (sin x)))) ...
;;

(defclass LispJIT-Blueprint ()
  ((op-func :initform nil :type symbol :accessor blueprint-op-func))
  (:documentation "
## [class] LispJIT-Blueprint

AbstractNodes which extends this class, is recognised as `LispJITAble` Node by Lisp-JIT-Compiler. This class possess information which is necessary for jit-compiling to cl code.
"))

;;(defmethod on-compiling-finalizing .. Trace and JIT and Return S exp.


(defmethod on-finalizing-compiling ((current-node LispJIT-Blueprint)
				    &rest variables)
  ;; (if (finalize-p?
  ;; ..
  )



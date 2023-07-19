
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

;; 2. Implementation
;;
;; On the end of calling of compile-forward-chain, a generic-function `on-finalizing-compiling` is invoked which users can append lisp-code as needed.
;; on-finalizing-compiling will give these informations:
;;   
;;     [TopLevel]
;;         |
;;     [SinNode1]   <- If invoked at this point...
;;         |
;;   [MoveTensorNode2]
;;         |
;;     [SinNode3]
;;         |
;;   [MoveTensorNode4]
;;         |
;;        ...
;; Info:
;;   - current-node SinNode1, to get corresponding LispJIT-Blueprint
;;   - past-variables (list TopLevel)
;;   - next-variables (list MoveTensorNode2), needed to judge whether accept node or not.
;;

(defparameter *operands* `(+ - * / move))

(defclass LispJIT-Blueprint ()
  ((operand :initform nil :type symbol :accessor blueprint-operand))
  (:documentation "
## [class] LispJIT-Blueprint

AbstractNodes which extends this class, is recognised as `LispJITAble` Node by Lisp-JIT-Compiler. This class possess information which is necessary for jit-compiling to cl code.
"))

;; pattern match tukaou kana...

(defmethod on-finalizing-compiling ((current-node LispJIT-Blueprint)
				    variable
				    next-variables)

  (print current-node)
  (print variable)
  (print next-variables)
  ;;(defmethod on-compiling-finalizing .. Trace and JIT and Return S exp.
  ;; (if (finalize-p?
  ;; .. oyogaseteoku
  ;; do-compile
  nil
  )


(defun test-case-tmp ()
  (with-devices (JITLispTensor cl-waffe2/backends.lisp:LispTensor)
    (let ((out (!sin (!add (randn `(10 10)) (randn `(10 10))))))
      (let ((res (build out)))
	res))))


;; defgeneric jitlisp-trace (operand (eq ...)) ...)



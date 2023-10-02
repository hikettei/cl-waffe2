
(in-package :cl-waffe2/backends.jit.cpu)

(defparameter *lazy-c-source* nil)
(defparameter *compiling-ntime-count* 0)

(defmethod on-finalizing-compiling ((device-name (eql 'JITCPUTensor)) iseq-fw iseq-bw)

  ;; [TODO]
  ;; define-impl but cached
  ;; arithmetic ops
  ;; op fusion by on-finalizing-compiling
  ;; メモリレイアウトの最適化 ... JITLispTensorの復活？
  ;; Compleするタイミング最適化
  (print *lazy-c-source*)

  (load-foreign-function *lazy-c-source*)

  (setf *lazy-c-source* "")
  (values iseq-fw iseq-bw))


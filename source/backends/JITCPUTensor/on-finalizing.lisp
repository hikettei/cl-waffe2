
(in-package :cl-waffe2/backends.jit.cpu)

(defparameter *caching-c-source* nil)
(defparameter *compiling-ntime-count* 0)

(defmethod on-finalizing-compiling ((device-name JITCPUTensor) iseq-fw iseq-bw)
  ;; [TODO]
  (values iseq-fw iseq-bw))


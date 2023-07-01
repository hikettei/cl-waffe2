
(in-package :cl-waffe2/vm.generic-tensor)

(defparameter *cache-directory* "~/.cache/cl-waffe2")

;; Caching Compiled fasl

;;
;; (defnode (SinNode ...) ) is named as [SinNode-CPUTensor-2D]
;; being defined as like:
;; (defun Sinnode-CpuTensor-2D (...)
;;    (optimize ...)
;;    (lambda (..) ...))
;; We store such like functions to reduce compile-time.
;;


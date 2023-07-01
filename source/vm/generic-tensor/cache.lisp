
(in-package :cl-waffe2/vm.generic-tensor)

;; Not used anymore?
(defparameter *cache-directory* "~/.cache/cl-waffe2")

;; The file cache.lisp provides an optimized kernel compiler for acceptor.lisp

;;
;; (defnode (SinNode ...) ) is named as [SinNode-CPUTensor-2D]
;; being defined as like:
;; (defun Sinnode-CpuTensor-2D (...)
;;    (optimize ...)
;;    (lambda (..) ...))
;; We store such like functions to reduce compile-time.
;;

;; 1. Create a LUT to store compiled-s-expression.

;; ==================================================================
;; A template form of compiled cl-waffe2 programs:
;; (labels ((SinNode-CPUTensor-3D () ;; Storeroom of compiled kernels used
;;             ...
;;           )
;;          (SinNode-CPUTensor-2D ()
;;             ...
;;           ))
;;
;; (forward/backward networks continues)
;; )
;; ==================================================================

(defvar *kernel-storeroom* nil "An storeroom to record all kernels used in compiling time.")

(defstruct Compiled-Kernel
  (name nil :type symbol) ;; SinNode-CPUTENSOR
  (body nil :type list)
  (view-identfier nil :type symbol)) ;; 2D 3D Flatten ...

(defun place-cached-kernels ()
  "(progn
;; [LUT PLACE]
(defun SinNode-CPUTensor-3D ()
    ...)

(defun SinNode-CPUTensor-2D ()
    ...)

...

;; Forward/Backward Process continues...
)"
  ;; Expand *kernel-storeroom*
  )

(defun use-kernel ()
  "An replace of (funcall fw-compiled)"
  )



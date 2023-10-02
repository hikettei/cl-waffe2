
(in-package :cl-user)

(defpackage :cl-waffe2/backends.jit.cpu
  (:documentation ":cl-waffe2/backends.jit.cpu provides JIT compiler from cl-waffe2 codes to well vectorized C codes.")
  (:use :cl
	:alexandria
	:cl-waffe2/distributions
	:cl-waffe2/vm
        :cl-waffe2/vm.generic-tensor
	:cl-waffe2/vm.nodes
        :cl-waffe2/base-impl)
  (:export
   #:*default-c-compiler*
   #:*compiler-flags*
   #:*viz-compiled-code*
   #:JITCPUTensor
   #:JITCPUScalarTensor
   #:cpujit-set-config
   #:with-cpu-jit))

(in-package :cl-waffe2/backends.jit.cpu)

;; Utils
(defun symb (&rest inputs)
  (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out)))))

(defun delete-newlines (string)
  (cl-ppcre:regex-replace-all #\newline string " "))

(defun c-name (string)
  (cl-ppcre:regex-replace-all "-" string "_"))

(defun range (from to)
  (loop for i fixnum upfrom from below to collect i))

(define-compiler-macro range (from to)
  `(loop for i fixnum upfrom ,from below ,to collect i))

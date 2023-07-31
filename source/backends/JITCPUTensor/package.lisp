
(in-package :cl-user)

;; TODO: Delete JITLispTensor

(defpackage :cl-waffe2/backends.jit.cpu
  (:documentation "from cl-waffe2 to C Compiler and oneDNN Support?")
  (:use :cl
        :cl-waffe2/distributions
        :cl-waffe2/vm.generic-tensor
        :cl-waffe2/vm.nodes
        :cl-waffe2/base-impl)
  (:export
   #:*default-c-compiler*
   #:JITCPUTensor
   #:JITCPUScalarTensor))

(in-package :cl-waffe2/backends.jit.cpu)

(defun compose (&rest fns)
  "fn_1(fn_2(fn_n...))"
  (if fns
      (let ((fn1 (car (last fns)))
            (fns (butlast fns)))
        #'(lambda (&rest args)
            (reduce #'funcall fns
                    :from-end t
                    :initial-value (apply fn1 args))))
      #'identity))

(defun symb (&rest inputs)
  (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out)))))

(defmacro with-ez-to-view (&body body)
  `(let ((*with-printing-tensor-omitted* t))
     ,@body))


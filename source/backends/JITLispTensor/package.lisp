
(in-package :cl-user)

(defpackage :cl-waffe2/backends.jit.lisp
  (:documentation "cl-waffe2/backends.jit.lisp demonstrates an example case of implementing jit with cl-waffe2.")
  (:use :cl
	:cl-waffe2/distributions
        :cl-waffe2/vm.generic-tensor
	:cl-waffe2/vm.nodes
        :cl-waffe2/base-impl)
  (:export
   #:JITLispTensor))


(in-package :cl-waffe2/backends.jit.lisp)


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



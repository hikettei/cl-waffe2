
(in-package :cl-waffe2/backends.metal)

(defun symb (&rest inputs)
  (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out)))))


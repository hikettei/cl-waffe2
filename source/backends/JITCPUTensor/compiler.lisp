
(in-package :cl-waffe2/backends.jit.cpu)

(defvar *compiled-code-buffer* nil "The variable collects generated C codes.")

(defmacro with-compiling-mode (&body body)
  "Initializes *compiled-code-buffer*"
  `(with-output-to-string (*compiled-code-buffer*)
     ,@body))

(defun write-buff (control-string &rest args)
  "Appends the given characters to *compiled-code-buffer*."
  (apply #'format *compiled-code-buffer* control-string args))

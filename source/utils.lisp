
(in-package :cl-waffe2)

(defmacro with-build ((forward variables parameters) out &body body)
  `(multiple-value-bind (,forward ,variables, parameters) (construct-forward ,out)
     ,@body))

;; (defmacro with-config :DTYPE, DEVICE, etc..)


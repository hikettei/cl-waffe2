
(cl:in-package :cl-user)

;; Usage: (asdf:load-system "aten")
(defpackage :cl-waffe2/backends.aten
  (:documentation "Aten works as a shader abstraction layer.")
  (:use :cl :cl-waffe2/base-impl :cl-waffe2/vm.nodes :cl-waffe2/vm.generic-tensor)
  (:export
   #:Aten
   #:Aten[Clang]
   #:Aten[Metal]))

(in-package :cl-waffe2/backends.aten)


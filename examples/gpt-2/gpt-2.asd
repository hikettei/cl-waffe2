
(in-package :cl-user)

(defpackage :gpt-2-asd
  (:use :cl :asdf :uiop))

(in-package :gpt-2-asd)

(defsystem :gpt-2-example
  :author "hikettei"
  :licence "MIT"
  :description "This is a sample project of inferencing GPT-2 on cl-waffe2"
  :pathname ""
  :serial t
  :depends-on (:cl-waffe2 :alexandria :numpy-file-format :jonathan :cl-ppcre)
  :components ((:file "package")
	       (:file "utils")
	       (:file "model")
	       (:file "tokenizer")))



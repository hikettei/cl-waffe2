

(in-package :cl-user)

(defpackage :cifar-asd
  (:use :cl :asdf :uiop))

(in-package :cifar-asd)

(defsystem :cifar10-sample
  :author "hikettei"
  :licence "MIT"
  :description "This is a sample project of training Cifar-10 with cl-waffe2"
  :pathname ""
  :serial t
  :depends-on (:cl-waffe2 :numpy-file-format)
  :components ((:file "package")
	       (:file "data-loader")
	       (:file "cnn")))





(in-package :cl-user)

(defpackage :mnist-asd
  (:use :cl :asdf :uiop))

(in-package :mnist-asd)

(defsystem :mnist-sample
  :author "hikettei"
  :licence "MIT"
  :description "This is a sample project of training MNIST with various models in cl-waffe2"
  :pathname ""
  :serial t
  :depends-on (:cl-waffe2 :numpy-file-format)
  :components ((:file "package")
	       (:file "data-loader")
	       (:file "cnn")
	       (:file "mlp")))





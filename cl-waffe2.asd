
(in-package :cl-user)

(defpackage :cl-waffe2-asd
  (:use :cl :asdf :uiop))

(in-package :cl-waffe2-asd)

(defsystem :cl-waffe2
  :author "hikettei"
  :licence "MIT"
  :description "Deep Learning Framework"
  :pathname "source"
  :serial t
  :components ((:file "vm/nodes/package")
	       (:file "vm/nodes/shape"
	       (:file "vm/nodes/node")
	       (:file "vm/nodes/conditions")
	       (:file "vm/nodes/defnode")
	      

	       ))

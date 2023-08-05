
(in-package :cl-user)

(defpackage :cl-waffe2-cnn-example
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/distributions
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl
   :cl-waffe2/nn))

(in-package :cl-waffe2-cnn-example)

(defsequence CNN ()
	     (Conv2D 3 6 `(5 5))
	     (asnode #'!relu)
	     (Conv2D 6 12 `(5 5))
	     (asnode #'!relu)
	     (asnode #'!reshape t (* 16 5 5))
	     (LinearLayer (* 16 5 5) 120)
	     (asnode #'!relu)
	     (LinearLayer 120 84)
	     (asnode #'!relu)
	     (LinearLayer 84 10))

(print (CNN))



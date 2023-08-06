
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

(defsequence Cifar10-CNN ()
	     (Conv2D 3 6 `(5 5))
	     (asnode #'!relu)
	     (MaxPool2D `(2 2))
	     (Conv2D 6 16 `(5 5))
	     (asnode #'!relu)
	     (MaxPool2D `(2 2))
	     (asnode #'!reshape t (* 16 22 22))
	     (LinearLayer (* 16 22 22) 120)
	     (asnode #'!relu)
	     (LinearLayer 120 84)
	     (asnode #'!relu)
	     (LinearLayer 84 10))

;; [TODO]
;; BugFix on !permute backward.
;; cl-waffe2/threads:scheduled-pdotimes (i 100) macro
;;

(defmethod train ((model Cifar10-CNN) x y)
  (!mean
   (softmax-cross-entropy
    (call model x)
    y)))

(print (Cifar10-CNN))



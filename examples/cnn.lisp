
(in-package :cl-user)

(defpackage :cl-waffe2-cnn-example
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/distributions
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl
   :cl-waffe2/nn
   :cl-waffe2/backends.jit.cpu
   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.lisp))

(in-package :cl-waffe2-cnn-example)

;; Parameters: https://www10.cs.fau.de/publications/theses/2022/Master_HolzmannMichael.pdf

(defsequence CNN ()
	     (Conv2D 3 16  `(3 3))
	     (asnode #'!relu)     
	     (MaxPool2D    `(2 2))
	     (Conv2D 16 32 `(5 5))
	     (asnode #'!relu)
	     (MaxPool2D `(2 2))
	     (asnode #'!reshape t (* 32 5 5)) 
	     (LinearLayer (* 32 5 5) 10))


;; [TODO]
;; BugFix on !permute backward.
;; cl-waffe2/threads:scheduled-pdotimes (i 100) macro
;;

(defmethod train ((model CNN) x y)
  (!mean
   (softmax-cross-entropy
    (call model x)
    y)))

(defun test ()
  (let ((model (build (train (CNN) (randn `(10 3 32 32)) (randn `(10 10))))))
    (time (forward model))))
    
(print (CNN))

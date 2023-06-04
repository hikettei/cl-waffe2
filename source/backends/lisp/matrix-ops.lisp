
(in-package :cl-waffe2/backends.lisp)

(define-impl (2dMatmulNode :device LispTensor)
	     :save-for-backward (t t nil)
	     :forward
	     ((self x y out)
	      )
	     :backward
	     ((self dout dx dy do)

	      ))

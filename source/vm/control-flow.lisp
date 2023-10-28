
(in-package :cl-waffe2/vm)

;; Provides two IRs:
;;  [Block]
;;    IfNode(Block1, Block2)
;;    MapNode(Block1, Block2)

;; [TODO] inside of *block* error messaging
#|
Not tested yet but this feature should be implemented in the future release
so ive decide to leave this code as it is.
(defparameter *block* nil)
(defun make-block (iseq)
  (let ((block-name (gensym "BLOCK")))
    (make-wfop
     #'(lambda ()
	 (let ((*block* block-name))
	   (accept-instructions iseq)))
     (wfop-self (car (last iseq)))
     #'(lambda ()
	 (with-output-to-string (out)
	   (format out "{~a~%" block-name)
	   (with-indent-to iseq
	     (dolist (i iseq)
	       (format out "    ~a~%" i)))
	   (format out "}")))
     nil)))
|#

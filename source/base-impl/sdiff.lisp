
(in-package :cl-waffe2/base-impl)

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Symbolic Diff
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; log_e(1+x)
(define-symbolic-path
    (!loge ((x)
	    (trivia:match x
	      ((or (list '!add var (or 1.0d0 1.0 1))
		   (list '!add (or 1.0d0 1.0 1) var)
		   (list '!+ var (or 1.0d0 1.0 1))
		   (list '!+ (or 1.0d0 1.0 1) var)
		   (list '!scalar-add (or 1.0d0 1.0 1) var))
	       var))))
    (:device cl-waffe2/backends.lisp::LispTensor)
    (x)
  `(!log1p ,x))




(in-package :cl-waffe2/vm.nodes)

;; [TODO] cl-waffe2.asd
;; [TODO] Docstring
;; [TODO] Test

;; for testing
(defun !!log (x) (log x))
(defun !!add (a b) (+ a b))
(defun !!+   (a b) (+ a b))

(defun !!log1p (x) (print "LOG1P") (log (+ 1 x)))

(define-symbolic-path (!!log
		       ((x)
			(trivia:match x
			  ((or (list '!!+ 1 var)
			       (list '!!+ var 1))
			   var))))			
    (:device cl-waffe2/backends.cpu:CPUTensor)
    (x)
  `(!!log1p ,x))

(defun hoge (x)
  (!!log (!!+ x 1)))
  
;; (!div X X)
;; (!mul (!div X X) (!div X X)
;; (!add (!exp X) (!exp X))

  ;; Kwargs?

(defparameter *enable-symbolic-path* T)

(defmacro define-symbolic-path ((subject &rest clauses) (&key (device t) (env (gensym))) (&rest form-binds) &body replacement &aux (pattern-names))
  "
## [macro] define-symbolic-path

```lisp

```

Reverved Words:

Local Patterns:

"

  (with-gensyms (args form result)
    `(define-compiler-macro ,subject (&whole ,form &rest ,args &environment ,env)
       (declare (ignorable ,env))
       (flet (,@(loop for c in clauses
		      collect
		      `(,(car (push (gensym) pattern-names)) ,@c)))
	 (let ((,result (list ,@(loop with pattern-names = (reverse pattern-names)
				      while pattern-names
				      for nth fixnum upfrom 0 collect
				      `(,(pop pattern-names) (nth ,nth ,args))))))
	   (if (every #'(lambda (x) x) ,result)
	       (multiple-value-bind (,@form-binds) (apply #'values ,result)
		 `(progn;;load-time-value
		   (locally (declare (notinline ,',subject))
		     (if (and *enable-symbolic-path*
			      (subtypep (car *using-backend*) ',',device))
			 (progn
			   ,,@replacement)
			 ,,form))))
	       ,form))))))


;; defpath ... !reluなどの関数名だけで指定できる

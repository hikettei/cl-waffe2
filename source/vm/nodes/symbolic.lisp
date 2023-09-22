
(in-package :cl-waffe2/vm.nodes)

;; [TODO] cl-waffe2.asd (OK)
;; [TODO] Docstring
;; [TODO] Test

(defparameter *enable-symbolic-path* T "
## [parameter] `*enable-symbolic-path*`

Indicates function calls replaced with `define-symbolic-path` effects on the result.

Set T to enable symbolic diff. In default: `T`.
")

(defmacro define-symbolic-path ((subject &rest clauses) (&key (device t) (env (gensym))) (&rest form-binds) &body replacement &aux (pattern-names))
  "
## [macro] define-symbolic-path

```lisp
(define-symbolci-path (subject &rest clause) (&key (device t) (env (gensym))) (&rest form-binds) &body replacement)
```

Defines a compiler-macro so-called **Symbolic Differentiation** which fuses several nodes into one, or replaces with another nodes. Sometimes, can combine cl-waffe2 functions to compose bad a computation node in tern of speed and safety; nodes (e.g: `(log (1+ x))`, `(log (exp x))`) should be represented as `(log1p x)` or `x` in the first place for reverse mode autodiff, and some nodes like `(!div X X)` should be deleted before compiling. This macro, however, enables that detecting such combinations and replacing them with another node before compiling.

First, describe `subject` the function name to be replaced (e.g.:`!log` `!sum` `!sin` etc...). And then, each `caluses` receive an argument of corresponding position, and determine if the form can be replaced or transformed. Plus, currently using devices can be included to the condition: Only after a car of `*using-backend*` is a subtype of specified `device`, symbolic path is replaced. At the last, each result of `clauses` will be binded to `form-binds`, and return the improved code at the `replacement` in a manner of `defmacro.` If needed, `&environment` is binded to the `env`.

### Inputs

- `clauses[form]` `((var) body)`

### Effects

- defines a compiler-macro named after `subject`.

### Example

Considering composing `(!!log (!!+ x 1))`

```lisp
(defun !!log (x)
  (print \"LOG\")
  (log x))

(defun !!+   (a b)
  (print \"+\")
  (+ a b))

(defun !!log1p (x)
  (print \"LOG1P\")
  (log (+ 1 x)))
```

```lisp
(define-symbolic-path (!!log
		       ((x)
			(trivia:match x
			  ((or (list '!!+ 1 var)
			       (list '!!+ var 1))
			   var))))			
    (:device cl-waffe2/backends.cpu:CPUTensor)
    (x)
  `(!!log1p ,x))
```

```lisp
(defun test-form (x)
  (!!log (!!+ 1 (!!log x))))
```

```lisp
(test-form 2)
;; LOG
;; LOG1P
;; 0.52658904
(test-form 2)
(setf *enable-symbolic-path* NIL) ;; Disable this feature
;; LOG
;; +
;; LOG
;; 0.52658904
```

Since the macro defines a compile-macro, this optimizing feature can be added one per one function.
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
;; こっちは一つのpathにつき複数のデバイスで登録できる
(defmacro define-symbolic-op-replace ())

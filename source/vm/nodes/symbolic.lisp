
(in-package :cl-waffe2/vm.nodes)

(defparameter *enable-symbolic-path* T "
## [parameter] `*enable-symbolic-path*`

Indicates function calls replaced with `define-symbolic-path` and `define-bypass` effects on the result.

Set T to enable symbolic diff. In default: `T`.
")

(defmacro define-symbolic-path ((subject &rest clauses) (&key (device t) (env (gensym))) (&rest form-binds) &body replacement &aux (pattern-names))
  "
## [macro] define-symbolic-path

```lisp
(define-symbolci-path (subject &rest clause) (&key (device t) (env (gensym))) (&rest form-binds) &body replacement)
```

Defines a compiler-macro so-called **Symbolic Differentiation** which fuses several nodes into one, or replaces with another nodes. Sometimes, can combine cl-waffe2 functions to compose bad a computation node in tern of speed and safety; nodes (e.g: `(log (1+ x))`, `(log (exp x))`) should be represented as `(log1p x)` or `x` in the first place for reverse mode autodiff, and some nodes like `(!div X X)` should be deleted before compiling. This macro, however, enables that detecting such combinations and replacing them with another node before compiling.

First, describe `subject` the function name to be replaced (e.g.:`!log` `!sum` `!sin` etc...). And then, each `caluses` receive an argument of corresponding position, and determine if the form can be replaced or transformed. Plus, currently using devices can be included to the condition: Only after a car of `*using-backend*` is a subtype of `device`, symbolic path is replaced. At the last, each result of `clauses` will be binded to `form-binds`, and return the improved code at the `replacement` in a manner of `defmacro.` If needed, `&environment` is binded to the `env`.

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

Since the macro defines a compile-macro, this optimizing feature can be added one per one function. For example, If the purpose is to replace the standard implementation of `cl-waffe2/nn:!relu` with another fused and device-specific implementation, use the `define-bypass` macro.
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

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *bypass-table* (make-hash-table)))

(defstruct Features-Table
  (key nil :type symbol)
  (replacement nil :type symbol))

(defun find-from-feature-table (device list)
  (find device list :test #'(lambda (x y) (eql y x)) :key #'features-table-key))

(defmacro define-bypass (device name replacement)
  "
## [macro] define-bypass

```lisp
(define-bypass device name replacement)
```

Defines a compiler-macro called **bypass**, which replaces an existing function call with another one. If you want to fuse nodes created by functions which creates computation node (e.g.: !relu !softmax !gelu), declare an alternative route with this function, and they can be replaced with like: ReLUNode, SoftmaxNode, GeLUNode.

The replacing is done when one of `*using-backend*` is the equivalent to `name[symbol]`, the funcall of `name[symbol]` will be replaced with `replacement[symbol]`. Note that before and after the replacement, they both should take the same arguments, same keywords. Unlike `define-symbolic-path`, there is no restriction of numbers that can be registered as a bypass to the single function; A single `!relu` can be replaced with: `!relu-cpu-fuse`, `!relu-cuda-fuse` for example.

### Example

```lisp
(defun !!relu (x)
  (print \"RELU\")
  x)

(defun !!relu-fuse (x)
  (print \"RELU_FUSE\")
  x)

(defun op (x)
  (!!relu x))

(define-bypass cl-waffe2/backends.cpu:CPUTensor !!relu !!relu-fuse)

(op 3)
;; RELU_FUSE
;; 3

(setf *enable-symbolic-path* nil)

(op 3)
;; RELU
;; 1
```

"  
  (eval-when (:compile-toplevel :load-toplevel :execute)
    (let ((first-one (null (gethash name *bypass-table*))))
      (with-gensyms (form args result)
	(if first-one
	    (progn
	      (setf (gethash name *bypass-table*) (make-hash-table))
	      (setf (gethash device (gethash name *bypass-table*)) (make-features-table :key device :replacement replacement))
	      `(define-compiler-macro ,name (&whole ,form &rest ,args)
		 `(let ((,',result (and *enable-symbolic-path*
					(find-from-feature-table (car *using-backend*) (hash-table-values (gethash ',',name *bypass-table*))))))
		    (progn;load-time-value
		      (locally (declare (notinline ,',name))
			(if ,',result
			    (funcall (features-table-replacement ,',result) ,@,args)
			    ,,form))))))
	    (progn
	      (setf (gethash device (gethash name *bypass-table*)) (make-features-table :key device :replacement replacement))
	      T))))))

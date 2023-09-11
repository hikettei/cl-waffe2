
(in-package :cl-waffe2)

;;
;; network.lisp: Here we provide an advanced APIs for formulating computation nodes; composing, tracing, converting etc...
;;

(defmodel (Encapsulated-Node (self node-func)
	   :slots ((node-func :initarg :node-func))
	   :initargs (:node-func node-func)
	   :documentation "(asnode ) dedicated Composite. Wraps the given node-func (excepted to construct networks) with no `:where` dependency"
	   :on-call-> ((self x)
		       (funcall (slot-value self 'node-func) x))))

(defmethod on-print-object ((model Encapsulated-Node) stream)
  (format stream "
    ~a
" (slot-value model 'node-func)))

(defmacro asnode (function &rest arguments)
  "
## [macro] asnode

```lisp
(asnode function &rest arguments)
```

Wraps the given `function` which excepted to create computation nodes with the `Encapsulated-Node` composite. That is, functions are regarded as a `Composite` and be able to use a variety of APIs (e.g.: `call`, `call->`, `defmodel-as` ...).

In principle, a function takes one argument and returns one value, but by adding more `arguments` the macro automatically wraps the function to satisfy it. For example, `(asnode #'!add 1.0) is transformed into: #'(lambda (x) (!add x 1.0))`. So the first arguments should receive AbstractTensor.

### Usage: call->

It is not elegant to use `call` more than once when composing multiple models.

```lisp
(call (AnyModel1)
      (call (AnyModel2)
             (call (AnyModel3) X)))
```

Instead, you can use the `call->` function:

```lisp
(call-> X
        (AnyModel1)
        (AnyModel2)
        (AnyModel3))
```

However, due to constrains of `call`, it is not possible to place functions here. `asnode` is exactly for this!

```lisp
(call-> X
        (AnyModel1)
        (asnode #'!softmax)
        (asnode #'!view 0) ;; Slicing the tensor: (!view x 0 t ...)
        (asnode #'!add 1.0) ;; X += 1.0
        (asnode !matmul Y) ;; X <- Matmul(X, Y)
        )
```

### Usage2: defmodel-as

The macro `cl-waffe2/vm.nodes:defmodel-as` is able to define new functions/nodes from existing `Composite`. However, this macro only needs the traced computation nodes information to do this. As the simplest case, compiling the AbstractNode `SinNode` (which is callable as `!sin`) into static function, `matrix-sin`.

```lisp
(defmodel-as (asnode #'!sin) :where (A[~] -> B[~]) :asif :function :named matrix-sin)

(matrix-sin (ax+b `(10 10) 0 1)) ;; <- No compiling overhead. Just works like Numpy
```

On a side note: `Encapsulated-Node` itself doesn't provide for `:where` declaration, but you can it with the keyword `:where`.
"
  (if arguments
      `(Encapsulated-Node #'(lambda (x) (funcall ,function x ,@arguments)))
      `(Encapsulated-Node ,function)))

(defmacro call-> (input &rest nodes)
  "
## [macro] call->

```lisp
(call-> input &rest nodes)
```

Starting from `input`, this macro applies a composed function.

```lisp
(call-> (randn `(3 3))       ;; To the given input:
	(asnode #'!add 1.0)  ;;  |
	(asnode #'!relu)     ;;  | Applies operations in this order.
	(asnode #'!sum))))   ;;  ↓
```

`nodes` could be anything as long as the `call` method can handle, but I except node=`Composite`, `AbstractNode`, and `(asnode function ...)`.
"
  (let ((nodes (reverse nodes)))
    (labels ((expand-call-form (rest-inputs)
	       (if rest-inputs
		   `(call ,(car rest-inputs)
			  ,(expand-call-form (cdr rest-inputs)))
		   input)))

      (expand-call-form nodes))))

;; Sequence Printer

(defgeneric sequencelist-nth (n sequence-model)
  (:documentation "
## [generic] sequencelist-nth
"))

(defmacro defsequence (name (&rest args) &optional docstring &rest nodes)
  "
## [macro] defsequence

```lisp
(defsequence (name (&rest args) &optional docstring &rest nodes))
```

Defines a Composite that can be defined only by the `call->` method.

### Inputs

`name[symbol]` defines the new Composite after `name`

`args[list]` a list of arguments that used to initialize nodes. Not for `call`.

`docstring[string]` docstring

### Example

```lisp
(defsequence MLP (in-features)
    \"Docstring (optional)\"
    (LinearLayer in-features 512)
    (asnode #'!tanh)
    (LinearLayer 512 256)
    (asnode #'!tanh)
    (LinearLayer 256 10))

;; Sequence can receive a single argument.
(call (MLP 786) (randn `(10 786)))
```

Tips: Use `(sequencelist-nth n sequence-model)` to read the nth layer of sequence.
"
  (let* ((nodes `(,docstring ,@nodes))
	 (documentation (if (stringp (car nodes))
			    (car nodes)
			    ""))
	 (nodes (if (stringp (car nodes))
		    (cdr nodes)
		    nodes))
	 (length   (length nodes))
	 (readers  (loop for i in nodes collect (gensym "Reader")))
	 (names    (loop for i in nodes collect (gensym "Composite")))
	 (keywords (loop for i in nodes collect (intern (symbol-name (gensym "KW")) "KEYWORD"))))

    `(progn
       (defmodel (,name (self ,@args)
		  :slots (,@(loop for name in names
				  for kw   in keywords
				  for reader in readers
				  collect `(,name :initarg ,kw :reader ,reader))
			  (length :initform ,length :reader sequencelist-length))
		  :initargs ,(let ((result))
			       (loop for kw in keywords
				     for node in nodes
				     do (push node result)
					(push kw result))
			       result)
		  :on-call-> ((self x)
			      (call-> x
				      ,@(loop for name in names
					      collect `(slot-value self ',name))))
		  :documentation ,documentation))
       
       (defmethod sequencelist-nth (n (model ,name))
	 (if (> n ,length)
	     (error "defsequence: ~a is out of range for ~a list." n ,length)
	     (funcall (nth n ',readers) model)))

       (defmethod on-print-object ((model ,name) stream &aux (length (sequencelist-length model)))
	 (format stream "
    <<~a Layers Sequence>>
" length)
	 (dotimes (i length)
	   (let ((layer (sequencelist-nth i model)))
	     (format stream "
[~a/~a]          ↓ 
~a"
		     (1+ i) length layer)))))))





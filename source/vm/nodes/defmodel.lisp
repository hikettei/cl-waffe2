
(in-package :cl-waffe2/vm.nodes)

;; Section: Creating a Module (A set of nodes).

(defclass Composite ()
  ((model-id :initform (gensym "W") :reader model-id)
   (traced?     :initform nil :type boolean :accessor composite-traced-p)
   (input-size  :initform nil :type list :accessor  composite-input-size)
   (output-size :initform nil :type list :accessor composite-output-size))
  (:documentation "Composite is a fundamental datatype for all neural network models. The name composite is so named because it is used to bundle computation nodes constructed by defnode.

In cl-waffe2, All models should be a subtype of this class, and shall return a forward propagation computation node using the **call** function.

In order to define your model with Composite, two methods are available.

### Extend Composite Class (Slightly Complicated)

First, define your class with extending Composite Class.

```lisp
(defclass LinearModel (Composite)
   ((weight ...) ; <- set parameters here.
    (bias   ...))
```

Second, define forwarrd step with overriding call method.

```lisp
(defmethod call ((model LinearModel) &rest inputs)
     ... )
```

It should work like:

```(call (make-instance 'LinearModel in-features out-features) args1 ...) ```

### Using defmodel macro

The defmodel macro simplifies the above redundant notation and also solves the problem that call can only use &rest as an argument. Therefore, I'm depcrecated with the method above, instead, use defmacro. For detailed usage, see the documentation of defmacro.
"))

(defgeneric call (model &rest inputs) (:documentation "All models in cl-waffe2, should implement this generic function. This generic function returns the computation node of the forward propagation of the model.

The generic function call is also used to step forward of AbstractNode, that is, works as if forward."))

(defmethod call :before ((model Composite) &rest inputs)
  (declare (ignore inputs))
  ;; Update States?
  (assert (subtypep (class-of model) 'Composite)
	  nil
	  "Assertion Failed with call method, because the model ~a isn't subtype of cl-waffe2/vm.nodes:Composite." model))

(defmethod call :around ((model Composite) &rest inputs)
  (let ((result (multiple-value-list (call-next-method))))

    ;; Traces the input/output tensor's shape.
    (setf (composite-input-size model)
	  (map 'list #'(lambda (x)
			 (when (typep x 'cl-waffe2/vm.generic-tensor:abstracttensor)
			   (shape x)))
	       inputs))
    
    (setf (composite-output-size model)
	  (map 'list #'(lambda (x)
			 (when (typep x 'cl-waffe2/vm.generic-tensor:abstracttensor)
			   (shape x)))
	       result))
    (setf (composite-traced-p model) t)
    
    result))

(defmethod call ((model AbstractNode) &rest inputs)
  (apply #'forward model inputs))


(defmacro define-forward-function (model forward-function)
  "forward-function = funcallable function"
  `(defmethod call ((model ,model) &rest inputs)
     (apply ,forward-function model inputs)))


;; Idea:
;; Automatically dispatching Conv1D Conv2D Conv3D ConvNd ... with generic
;; Add: ModuleList

(defgeneric on-print-object (model stream) (:documentation "

## [generic] on-print-object

```
(on-print-object model stream)
```

Every time the composite is rendered, this function is called.

```
<Composite: NAME{...}(
    [...] <- The content of here is depends on on-print-object
    [PARAMTETERS]
)
```"))

(defmethod on-print-object ((model Composite) stream))

;; Enhancement: Inference the size of Input/Output in advance.
(defmethod on-print-object :after ((model Composite) stream)
  (when (composite-traced-p model)
    (format stream "
    <Input : ~a -> Output: ~a>
" (composite-input-size model)
  (composite-output-size model))))
    
(defmacro defmodel ((name
		     (self-name &rest constructor-arguments)
		     &key
		       (slots nil)
		       (initargs)
		       (on-call-> nil)
		       (documentation ""))
		    &body constructor-body)
  "
```
(defmodel ((name
	     (self-name &rest constructor-arguments)
		      &key
		       (slots nil)
		       (initargs)
		       (on-call-> nil)
		       (documentation \"\"))
		    &body constructor-body)
```

defmodel is a macro used to describe the model of neural network with `Composite` class.

### Effects

   1. defines a class named **name**

   2. defines a function named **name** with the constructor-arguments and constructor-body.


### Inputs

  1. name[Symbol]  All models, and constructors for the model, are named after it.
  2. (self-name &rest constructor-arguments)
    The constructor function is defined as:
    (defun ,name (self-name ,@constructor-arguments)
       ...)

  3. slots ((slot-option1) (slot-option2) ...)
    Parameters of the inherited Composite class. It has the same syntax as defclass slots

  4. initargs (:accessor-name1 accessor-init-form1 :accessor-name2 accessor-init-form2 ...
    Unlike CL's structure, classes are tend to rebundant when writing the process of initializing slots. To make this simple, this argument was introduced. It works like a structure's constructor!

  5. documentation[String]

  6. `on-call->` [One of: nil symbol-name function list]
     on-call-> is used to control the behaviour of *call* function.

  7. `on-print-object` [null or body]

### Example

```lisp
(defmodel (ExampleLayer (self features)
               ;; Options/Utils Here,
               :slots    ((param :initarg :param))
               :initargs (:param (make-tensor `(,features) :requires-grad t))
               :documentation \"ExampleLayer is a ...\")

    ;; After make-instance is called, the form below is called.
    ;; make-instance -> make-instance :after -> this form.

    (print self)     ;; <- Initialized ExampleLayer
    (print features) ;; <- constructor-arguments are also used here.
    (print \"ExampleLayer is created!\"))

;; The model you created, works like:
(let ((layer (ExampleLayer 10)))
    (call layer ...))
```

### Describe Forward Propagation

The option `on-call->` can control the behaviour of *call* function.

`on-call->` could be one of these case:

First case,  `on-call->` is nil:

  cl-waffe2 calls the **call** function when doing forward propagation of the model.

Second case, `on-call->` is symbol-name:

   cl-waffe2 calls the specified function at on-call-> parameter, when doing forward propagation of the model.

   symbol-name could be also one of: method's name function's name.

   For example, set `:on-call-> = call-example-layer` which defined as:

```lisp
   (defmethod call-example-layer ((model ExampleLayer) x y)
       (print \"call-example-layer is used!\")
       ...)
```


```lisp
   (call (ExampleLayer 10) tensor) ;; call-example-layer is used!
```

   (Complex model assignments like ConvND, for example, can be achieved by assigning generic function names to symbols.)

[Third case] `on-call->` is function (i.e.: lambda):

   cl-waffe2 calls the given lambda function as a forward propagation.

[Fourth case] `on-call->` is a list:

   The List, should be this format.

   `((arguments) body)`

   This argument is expanded into `#'(lambda ,@on-call->)` and works as well as 3."
  (declare (type (or symbol function list null) on-call->))
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (progn
       ;; E.g.: The case when we want to define LinearLayer Model...
       ;; defines LinearLayer class
       (defclass ,name (Composite)
	 (,@slots)
	 (:documentation ,(format nil "class, ~a" documentation)))

       ;; Creates a constructor named (linearlayer constructor-arguments)
       (defun ,name (,@constructor-arguments)
	 ,(format nil "An constructor for model.")
	 (let ((,self-name (make-instance
			    ',name
			    ,@initargs)))
	   ,@constructor-body
	   ,self-name))

       ;; Registers forward funcition cl-waffe2 uses.
       ;; on-call-> could be one of them:
       ;; 1. method-name
       ;; 2. function-name
       ;; 3. compiled-function
       ;; 4. list (compiled automatically)
       ,(typecase on-call->
	  (null nil)
	  (symbol   `(define-forward-function ,name #',on-call->))
	  (function `(define-forward-function ,name ,on-call->))
	  (list     `(define-forward-function ,name #'(lambda ,@on-call->))))
       t)))

(defmethod find-params ((model Composite))
  (let ((names (map 'list #'c2mop:slot-definition-name (c2mop:class-slots (class-of model)))))
    (loop for name in names
	  if (and (subtypep (class-of (slot-value model name))
			    'cl-waffe2/vm.generic-tensor:AbstractTensor)
		  (slot-value (slot-value model name) 'cl-waffe2/vm.generic-tensor:requires-grad))
	    collect (cons name (slot-value model name)))))

(defmethod render-model-content ((model Composite))
  (let* ((parameters (find-params model))
	 (longest-name (1+ (loop for p in parameters
				 maximize (length (symbol-name (car p)))))))
    
    (with-output-to-string (out)
      (format out "(")
      (on-print-object model out)
      (if parameters (format out "~%"))
      (dolist (p parameters)
	(let* ((name  (symbol-name (car p)))
	       (param (cdr p))
	       (name-len (length name)))
	  (format out "    ~a" name)
	  (dotimes (i (- longest-name name-len)) (princ " " out))
	  (format out "-> ~a~%" (shape param))))
      (format out ")"))))

(defmethod print-object ((model Composite) stream)
  (format stream
	  "<Composite: ~a{~a}~a>"
	  (class-name (class-of model))
	  (model-id model)
	  (render-model-content model)))


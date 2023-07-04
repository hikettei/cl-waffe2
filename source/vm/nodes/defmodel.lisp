
(in-package :cl-waffe2/vm.nodes)

;; Section: Creating a Module (A set of nodes).

(defclass Composite ()
  ((model-id :initform (gensym "W") :reader model-id)
   (subscript-linter :initform nil :initarg :linter-f :reader composite-linter-f)
   (linter-state1 :initform nil :initarg :linter-state1 :reader composite-linter1)
   (linter-state2 :initform nil :initarg :linter-state2 :reader composite-linter2)
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
    
    (apply #'values result)))

(defmethod call ((model AbstractNode) &rest inputs)
  (apply #'forward model inputs))

(defmethod call ((model cl-waffe2/vm.generic-tensor:Compiled-Composite) &rest inputs)
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

(defmethod composite-update-io ((model Composite) p1 p2)
  (declare (type function p1 p2))

  )

(defmethod composite-error ((model Composite) error-content)
  (shaping-error "Shaping-Error is detected when calling Composite.

~a

Here's a list of reports:

~a"
		 model
		 (with-output-to-string (out)
		   (loop for nth upfrom 1
			 for c in error-content
			 do (format out "~a. ~a~%"
				    nth c)))))

(defun preprocess-batch-symbol (p1)
  "~ -> GENSYM:XXX"
  (let ((name (gensym "~")))
    (values
     name
     (map-tree
      #'(lambda (x)
	  (typecase x
	    (symbol
	     (if (symbol-eq '~ x)
		 name
		 x))
	    (T x)))
      p1))))

(defun restore-symbol (sym out)
  "GENSYM:XXX -> ~"
  (map-tree
   #'(lambda (x)
       (typecase x
	 (symbol
	  (if (symbol-eq sym x)
	      '~
	      x))
	 (T x)))
   out))

(defmethod composite-where ((model Composite) inputs)
  "
## [function] composite-where

Predicts next output given inputs.

### Inputs

`inputs` ... nil or list

### Outputs

(values input output) or nil
"

  ;; At first: symbols
  ;; (funcall Linter-Function model nil   input-state1 input-state2)
  ;; Fixnums
  ;; (funcall Linter-Function model shape input-state1 input-state2)
  
  (with-slots ((linter-function subscript-linter)
	       (state1 linter-state1)
	       (state2 linter-state2))
      model
    
    (if linter-function
	(multiple-value-bind (sym1 state1) (preprocess-batch-symbol (car state1))
	  (multiple-value-bind (res in) (funcall linter-function model inputs state1 state1)
	    (values
	     (restore-symbol sym1 res)
	     (restore-symbol sym1 in))))
	nil)))

(defmacro defmodel ((name
		     (self-name &rest constructor-arguments)
		     &key
		       (slots nil)
		       (initargs)
		       (where nil)
		       (on-call-> nil)
		       (documentation ""))
		    &body constructor-body
		    &aux
		      (subscript-p1 (gensym "sub1"))
		      (subscript-p2 (gensym "sub2"))
		      (test-subscript-p (gensym "sub"))
		    
		      (inputs       (gensym "Inputs"))
		      (inputs1      (gensym "Inputs"))
		      (inputs2      (gensym "Inputs"))

		      (input-size (gensym "IO"))
		    
		      (try-out (gensym))
		      (try-err (gensym))
		      (try-rank-error (gensym))
		      (self-place1 (gensym)))
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

   (Complicated model assignments like ConvND, for example, can be achieved by assigning generic function names to symbols.)

[Third case] `on-call->` is function (i.e.: lambda):

   cl-waffe2 calls the given lambda function as a forward propagation.

[Fourth case] `on-call->` is a list:

   The List, should be this format.

   `((arguments) body)`

   This argument is expanded into `#'(lambda ,@on-call->)` and works as well as 3."
  (declare (type (or symbol function list null) on-call->))
  (let ((use-linter-p (not (null where))))
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       ;; E.g.: The case when we want to define LinearLayer Model...
       ;; defines LinearLayer class
       (defclass ,name (Composite)
	 (,@slots)
	 (:documentation ,(format nil "
## [model] ~a

```
(~(~a~)~a)
```

### Description

~a
"
				  (symbol-name name)
				  (symbol-name name)
				  (with-output-to-string (out)
				    (dolist (arg constructor-arguments)
				      (princ " " out)
				      (format out "~a" arg)))
				  documentation)))

       (defmethod read-where ((model ,name))
	 ',where)
       
       ;; Creates a constructor named (linearlayer constructor-arguments)
       (defun ,name (,@constructor-arguments)
	 ,(format nil "
## [function] ~a

An constructor function for ~a."
		  name
		  name)
	 (let ((,subscript-p1 ,(if use-linter-p
				   `(multiple-value-list (subscript ,where :allow-symbol t :constructor-args ,constructor-arguments))))
	       (,subscript-p2 ,(if use-linter-p
				   `(multiple-value-list (subscript ,where :fixed :t :allow-symbol t :constructor-args ,constructor-arguments)))))
	   (declare (ignorable ,subscript-p1 ,subscript-p2))
	   (labels ((,test-subscript-p (,self-place1 ,inputs ,inputs1 ,inputs2)
		      ;; inputs  = 
		      ;; inputs1 = 
		      ;; inputs2 = 
		      (declare (ignorable ,self-place1 ,inputs ,inputs1 ,inputs2))
		      ,(if use-linter-p
			   `(multiple-value-bind (,try-out ,try-err ,try-rank-error ,input-size)
				 (funcall (car ,subscript-p2) (or ,inputs ,inputs2))
			      (if ,try-rank-error
				  (multiple-value-bind (,try-out ,try-err ,try-rank-error ,input-size)
				      (funcall (car ,subscript-p1) (or ,inputs ,inputs1))
				    (declare (ignore ,try-rank-error))
				    (if (null ,try-err)
					(values ,try-out ,input-size)
					(composite-error ,self-place1 ,try-err)))
				  (if (null ,try-err)
				      (values ,try-out ,input-size)
				      (composite-error ,self-place1 ,try-err)))))))
	     (let ((,self-name (make-instance
				',name
				:linter-f
				#',test-subscript-p
				:linter-state1
				(fourth ,subscript-p1)
				:linter-state2
				(fourth ,subscript-p2)
				,@initargs)))
	       (declare (ignorable ,self-name))
	       ;; Update IO size

	       (multiple-value-bind (result input)
		   (composite-where ,self-name nil)
		 (when result
		   (setf (composite-traced-p ,self-name) t)
		   (setf (composite-input-size  ,self-name) input)
		   (setf (composite-output-size ,self-name) result)))
	       
	       ,@constructor-body
	       ,self-name))))

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

(defun composite-symbol-names (composite)
  (multiple-value-bind (in out) (parse-subscript (read-where composite))
    (values in out)))

(defun composite-input-tensor (composite ~
			       &key
				 (dtype :float)
				 (order :column)
				 (scalar-p nil))
  "Returns (make-input)"
  (declare (type Composite composite)
	   (type list ~))
  (let ((input-shape (composite-input-size composite)))
    (loop for i upfrom 0
	  for x in input-shape
	  collect (make-input (where-arg->shape ~ x)
			      (->keyword (nth-subscript i))
			      :scalar-p scalar-p
			      :dtype dtype
			      :order order))))

(defun where-arg->shape (~ shape)
  (flatten
   (loop for s in shape
	 if (symbol-eq s '~)
	   collect ~
	 else
	   collect s)))


(defun shape-compatible? (composite &rest inputs)
  "
## [function] shape-compatible?

Returns t if inputs are compatible with given composite, otherwise return an error.

Inputs: An list of input tensors
Return: (values output-shape input-shape-determined)

"
  (let ((linter-function (composite-linter-f composite))
	(inputs (map 'list #'shape inputs)))
    (funcall linter-function composite inputs inputs inputs)))

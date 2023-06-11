
(in-package :cl-waffe2/vm.nodes)

;; Section: Creating a Module (A set of nodes).

(defclass Composite ()
  nil
  (:documentation "Composite is a fundamental datatype for all neural network models. The name composite is so named because it is used to bundle computation nodes constructed by defnode.

In cl-waffe2, All models should be a subtype of this class, and shall return a forward propagation computation node using the **call** function.

In order to define your model with Composite, two methods are available.

1. Extend Composite Class (Slightly Complicated)
  First, define your class with extending Composite Class.

  (defclass LinearModel (Composite)
     ((weight ...) ; <- set parameters here.
      (bias   ...))
   Second, define forwarrd step with overriding call method.
   (defmethod call ((model LinearModel) &rest inputs)
     ... )

   It should work like:
   (call (make-instance 'LinearModel in-features out-features) args1 ...)

2. Using defmodel macro
   The defmodel macro simplifies the above redundant notation and also solves the problem that call can only use &rest as an argument. Therefore, I'm depcrecated with the method above, instead, use defmacro. For detailed usage, see the documentation of defmacro.
"))

(defgeneric call (model &rest inputs) (:documentation "All models in cl-waffe2, should implement this generic function. This generic function returns the computation node of the forward propagation of the model."))

(defmethod call :before ((model Composite) &rest inputs)
  (declare (ignore inputs))
  ;; Update States?
  (assert (subtypep (class-of model) 'Composite)
	  nil
	  "Assertion Failed with call method, because the model ~a isn't subtype of cl-waffe2/vm.nodes:Composite." model))

(defmacro define-forward-function (model forward-function)
  "forward-function = funcallable function"
  `(defmethod call ((model ,model) &rest inputs)
     (apply ,forward-function model inputs)))


;; Idea:
;; Automatically dispatching Conv1D Conv2D Conv3D ConvNd ... with generic
;; Add: ModuleList

(defmacro defmodel ((name
		     (self-name &rest constructor-arguments)
		     &key
		       (slots nil)
		       (initargs)
		       (on-call-> nil)
		       (documentation ""))
		    &body constructor-body)
  "defmodel is a macro used to describe the model of neural network with Composite class.

Inputs:
  - name[Symbol]
    All models, and constructors for the model, are named after it.
  - (self-name &rest constructor-arguments)
    The constructor function is defined as:
    (defun ,name (self-name ,@constructor-arguments)
       ...)

  - slots ((slot-option1) (slot-option2) ...)
    Parameters of the inherited Composite class. It has the same syntax as defclass slots

  - initargs (:accessor-name1 accessor-init-form1 :accessor-name2 accessor-init-form2 ...
    Unlike CL's structure, classes are tend to rebundant when writing the process of initializing slots. To make this simple, this argument was introduced. It works like a structure's constructor!

  - documentation[String]

  - on-call-> [One of: nil symbol-name function list]
    on-call-> is used to control the behaviour of *call* function.
# Declare The structure of model.
Example:
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

### Describe Forward Propagation

The option on-call-> can control the behaviour of *call* function.

on-call-> could be one of these case:

1. on-call-> is nil
   cl-waffe2 calls the **call** function when doing forward propagation of the model.

2. on-call-> is symbol-name
   cl-waffe2 calls the specified function at on-call-> parameter, when doing forward propagation of the model.
   symbol-name could be also one of: method's name function's name.

   For Example: set :on-call-> = call-example-layer which defined as:
   (defmethod call-example-layer ((model ExampleLayer) x y)
       (print \"call-example-layer is used!\")
       ...)

   (call (ExampleLayer 10) tensor) ;; call-example-layer is used!

   (Complex model assignments like ConvND, for example, can be achieved by assigning generic function names to symbols.)

3. on-call-> is function (i.e.: lambda)
   cl-waffe2 calls the given lambda function as a forward propagation.

4. on-call-> list
   The List, should be this format.
   ((arguments) body)
   This argument is expanded into #'(lambda ,@on-call->) and works as well as 3.


With regard to practical usage, visit my tutorial.
                "
  (declare (type (or symbol function list null) on-call->))
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (prog1
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
	  (list     `(define-forward-function ,name #'(lambda ,@on-call->)))))))


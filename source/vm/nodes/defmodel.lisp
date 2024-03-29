
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
  (:documentation "

## [class] Composite

Its `call` bundles several AbstractNode. It is not only used to represent a neural network but also convert nodes into functions or abstractnodes. You can forward composites with `(call composite arg1 arg2...)`. For the most case, composites are defined by the defmodel macro.

### [generic] on-print-object

```
(on-print-object model stream)
```

This generic function is used to customize how the model is printed on the display.

```
<Composite: NAME{...}(
    [...] <- The content of here is depends on on-print-object
    [PARAMTETERS]
)
```
"))

(defgeneric call (model &rest inputs) (:documentation "

```lisp
(call model &rest inputs)
```

`call` is a generic function which is used to `:forward`/`:on-call->` forms for an `AbstractNode`/`Composite` class respectively."))

(defmethod call :before ((model Composite) &rest inputs)
  ;; Update States?

  (let ((linter-p (read-where model)))
    (when linter-p
      (apply #'shape-compatible? model inputs)))
  
  (assert (subtypep (class-of model) 'Composite)
	  nil
	  "Assertion Failed with call method, because the model ~a isn't subtype of cl-waffe2/vm.nodes:Composite." model))

(defmethod call :around ((model Composite) &rest inputs)
  (let ((result (multiple-value-list (call-next-method))))


    ;; Finding all parameters belong to model, attribute following information:
    ;;  1. which models belongs to
    ;;  2. which slots  belongs to
    (update-tensor-state-dicts! model)
    
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
```
"))

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

;; Update?
(defmethod composite-error ((model Composite) error-content inputs outputs)
  (shaping-error "~a" (build-shape-error :call model (read-where model) inputs outputs error-content)))

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
	  (multiple-value-bind (res in) (funcall linter-function model nil inputs state1 state1)
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

		      (called-tensors (gensym))
		      (try-out (gensym))
		      (try-err (gensym))
		      (try-rank-error (gensym))
		      (self-place1 (gensym)))
  "
## [macro] defmodel
```
(defmodel ((name
	     (self-name &rest constructor-arguments)
		      &key
		       (slots nil)
		       (initargs)
                       (where nil)
		       (on-call-> nil)
		       (documentation \"\"))
		    &body constructor-body)
```

Defines a composite named `name`, and constructor function which also named `name` and receives `constructor-arguments` as arguments. The main process of its forward process is described in the `on-call->` slots.

### Inputs

`name[Symbol]` the macro defines an class and constructor function named after it.

`(self-name &rest constructor-arguments)` An initializer form of `constructor function`.

`slots ((slot-option1) (slot-option2) ...)` Parameters of the inherited Composite class. It has the same syntax as defclass slots`

`initargs (:accessor-name1 accessor-init-form1 :accessor-name2 accessor-init-form2 ...)` Unlike structures, CLOS classes are somewhat more cumbersome to initialise parameters. To make this process simple, put here initializer forms in advance likewise we do `(make-instance class-name ...)`.

`documentation[String]`

`on-call-> [One of: nil symbol-name function list]` The main proces of its forward process, later called with `(call model ...)` method. This method must be continuous from the given arguments.

`where[Subscript DSL] (Optional)` If you're planning to use `defmodel-as` macro, this form is needed.

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

```lisp
(defmodel (Softmax-Model (self)
	   :where (X[~] -> [~])
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (let* ((x1 (!sub x (!mean x  :axis 1 :keepdims t)))
	                      (z  (!sum   (!exp x1) :axis 1 :keepdims t)))
                           (!div (!exp x1) z)))))

;; Keep Using Lazily...
(proceed (call (Softmax-Model) (randn `(10 10)))
{CPUTENSOR[float] :shape (10 10) :named ChainTMP33497 
  :vec-state [computed]
  ((0.04800622   0.118814774  0.050377533  ~ 0.053051848  0.050124187  0.25575548)                    
   (0.15909052   0.11368358   0.12642372   ~ 0.114795394  0.033397682  0.07605342)   
                 ...
   (0.035624444  0.24828684   0.109363265  ~ 0.020787988  0.027314318  0.04515641)
   (0.030307569  0.24117047   0.03900468   ~ 0.014522874  0.036584295  0.0971196))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}

(defmodel-as (Softmax-Model) :asif :function :named softmax-static)

;; No compiling overhead
(softmax-static (randn `(10 10)))

{CPUTENSOR[float] :shape (10 10) :named ChainTMP33788 
  ((0.16722792   0.018530384  0.014159603  ~ 0.035353966  0.06128503   0.13559735)                    
   (0.14498742   0.11881006   0.0692616    ~ 0.03911829   0.10358454   0.02131605)   
                 ...
   (0.055657785  0.44042623   0.030706322  ~ 0.11048273   0.0097645    0.11959953)
   (0.059088983  0.11067564   0.120767005  ~ 0.15042976   0.06570089   0.20548664))
  :facet :input
  :requires-grad NIL
  :backward NIL}
```

### Dispatching on-call-> method

- `on-call-> is nil` In that case, users must define the call definiton manually like `(defmethod call ((model YourComposite) arg1 arg2) ...)`.

- `on-call-> is symbol` In that case, the composite invokes the method named `symbol` when call is invoked.

```lisp
;; Set :on-call-> call-example-layer


(defmethod call-example-layer ((model ExampleLayer) x y)
    (print \"call-example-layer is used!\"))
```

```lisp
(call (ExampleLayer 10) tensor) ;; call-example-layer is used!
```

### `on-call->` is a list

Directly defines a `call` method. Arguments must be: `(self arg1 arg2...)`
```lisp
...
    :on-call-> ((self x)
                (!sin x))
```
"
  (declare (type (or symbol function list null) on-call->))
  (let ((use-linter-p (not (null where)))
	(initargs-first (collect-initarg-slots slots constructor-arguments)))
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

~a
### Description

~a
"
				  (symbol-name name)
				  (symbol-name name)
				  (with-output-to-string (out)
				    (dolist (arg constructor-arguments)
				      (princ " " out)
				      (format out "~a" arg)))
				  (if where
				      (format nil "~%which transformation of shapes are defined as:~%```~%~a~%```" where)
				      "")
				  documentation)))

       (defmethod read-where ((model ,name)) ',where)
       
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
	   ;; Todo: Refactoring -> test-subscript-p
	   ;; test-subscript-p is only used to trace computation node
	   ;; Broadcasting isn't subject to consider.
	   (labels ((,test-subscript-p (,self-place1 ,called-tensors ,inputs ,inputs1 ,inputs2)
		      ;; inputs  = 
		      ;; inputs1 = 
		      ;; inputs2 =
		      (declare (ignorable ,self-place1 ,called-tensors ,inputs ,inputs1 ,inputs2))
		      ,(if use-linter-p
			   `(multiple-value-bind (,try-out ,try-err ,try-rank-error ,input-size)
				(funcall (car ,subscript-p2) (or ,inputs ,inputs2))
			      (if ,try-rank-error
				  (multiple-value-bind (,try-out ,try-err ,try-rank-error ,input-size)
				      (funcall (car ,subscript-p1) (or ,inputs ,inputs1))
				    (declare (ignore ,try-rank-error))
				    (if (null ,try-err)
					(values ,try-out ,input-size)
					(composite-error ,self-place1 ,try-err ,called-tensors ,try-out)))
				  (if (null ,try-err)
				      (values ,try-out ,input-size)
				      (composite-error ,self-place1 ,try-err ,called-tensors ,try-out)))))))
	     (let ((,self-name (make-instance
				',name
				:linter-f
				#',test-subscript-p
				:linter-state1
				(fourth ,subscript-p1)
				:linter-state2
				(fourth ,subscript-p2)
				,@(loop for slot in initargs-first
					     if slot
					       collect (intern (symbol-name (nth (1+ (position :initarg slot)) slot)) "KEYWORD")
					     if slot
					       collect (car slot))
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

(defmethod update-tensor-state-dicts! ((model Composite))
  (let ((parameters (find-params model)))
    ;; Parameters ... (cons name parameter)
    (loop for parameter-pair in parameters do
      (setf (tensor-state-dict-name (cdr parameter-pair)) (car parameter-pair)
	    (tensor-param-belongs-to (cdr parameter-pair)) model))
    T))

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
				 (input-shape nil) ;; set any list if composite doesn't have any
				 (inputs nil)
				 (dtype :float)
				 (order :column)
				 (argument-names nil)
				 (scalar-p-list nil))
  "Returns an list of InputTensor which is used to trace the computation nodes."
  (declare (type Composite composite)
	   (type list ~)
	   (ignore ~))
  (flet ((read-state (state nth)
	   (if (keywordp state)
	       state
	       (nth nth state)))
	 (read~ (~ nth)
	   (if (listp (car ~))
	       (nth nth ~)
	       ~)))
    #'read~
    
    (let ((input-shape (or input-shape (composite-input-size composite))))
      (loop for i upfrom 0
	    for x in input-shape
	    collect
	    ;; Hmm, in order to reuse compiled kernel, (shape (nth i inputs)) isn'g the best choise...
	    (let ((res (make-input (shape (nth i inputs));;(where-arg->shape (read~ ~ i) x)
				   (or (nth i argument-names) (->keyword (nth-subscript i)))
				   :create-from (nth i inputs)
				   :scalar-p (read-state scalar-p-list i)
				   :dtype (read-state dtype i)
				   :order order)))
	      (setf (tensor-protect-me res) t)
	      res)))))

(defun where-arg->shape (~ shape)
  (flatten
   (loop for s in shape
	 if (symbol-eq s '~)
	   collect ~
	 else
	   collect s)))


(defun shape-compatible? (composite &rest inputs1)
  "
## [function] shape-compatible?

Returns t if inputs are compatible with given composite, otherwise return an error.

Inputs: An list of input tensors
Return: (values output-shape input-shape-determined)

"
  (let ((linter-function (composite-linter-f composite))
	(inputs (map 'list #'shape inputs1)))
    (funcall linter-function composite inputs1 inputs inputs inputs)))


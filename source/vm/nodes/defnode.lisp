
(in-package :cl-waffe2/vm.nodes)


;;  [Overview of Computation Node in cl-waffe2]
;; ====================================================================================
;; Defining AbstractNode
;;  defnode - Declares AbstractDefinition of AbstractNode
;;       L define-impl     implements abstractnode as a macro to inline call-with-view, later (compile nil body) is called.
;;       L define-impl-op  implements abstractnode as a λfunction
;;
;;  define-op = defnode + define-impl-op

;; Defining Composite (Model)
;;  defmodel - Defines a new Composite
;;     L (Utils) defmodel-as Converts Composite into a function/AbstractNode
;; ====================================================================================

(defparameter *facet-monopoly-mode* nil "This parameter is used to ensure that all the calculations are performed under the same node. If this parameter is t, only use devices with Priority1, otherwise an error will occur.")

(defparameter *node-reject-case-table* (make-hash-table))

(defgeneric on-finalizing-compiling
    (current-node variable next-variable compile-me)
  (:documentation
   "
## [generic] on-finalizing-compiling

```lisp
(on-finalizing-compiling current-node variable next-variable compile-me)
```

The generic function `on-finalizing-compiling` is invoked after the body of `define-impl` is expanded when performing `compile-chain-forward`.

Return S expression to be embodied in the compiled code if needed, especially, devices which support jit-compiling will need this, because you can get information before and after the node.

### Inputs

```lisp
     [TopLevel]
         |
     [CosNode1]
         |
     [SinNode1]   <- If invoked at this point...
         |
   [MoveTensorNode2]
         |
     [SinNode3]
         |
   [MoveTensorNode4]
         |
        ...
```

`current-node (i.e.: SinNode1)` used to dispatch methods.

`variable (i.e.: corresponding variable of SinNode1)` returns the corresponding var of current node.

`next-variables (i.e.: corresponding variable of MoveTensorNode2)` returns corresponding variable of next node.

`compile-me[boolean]` If t, cl-waffe2 needs compiled code that works instantly.

See also: `the implementation of JITLispTensor`.
"))

;; [TODO] Add to docs.
(defgeneric on-finished-compiling (current-node)
  (:documentation "
## [generic] on-finished-compiling

```lisp
(on-finished-compiling current-node)
```

The method on-finished-compiling is once called when the node was reached the end.

### Example

```lisp
(defmethod on-finished-compiling ((current-node (eql 'JITCPUTensor)))
   ...
   )
```
"))

(defmethod on-finalizing-compiling ((current-node AbstractNode) variable next-variables compile-me)
  (declare (ignore variable next-variables))
  (when (next-method-p)
    (call-next-method)))

(defmethod on-finished-compiling ((current-node t))
  (when (next-method-p)
    (call-next-method)))

(defun node-compatible-p (node-name inputs)
  (declare (type list inputs))
  "the node is adopted when:
reject-when=nil, or (apply reject-when inputs)=t"
  (let ((table (gethash node-name *node-reject-case-table*)))
    (or (null table)
	(not (apply (the function table) inputs)))))

(defun set-node-reject-case (node-name function)
  (when function
    (setf (gethash node-name *node-reject-case-table*) function)))

;; Is it ok?
(defun env-parameter-p (sym) (equal (aref (symbol-name sym) 0) #\&))

;; FixME there must be much clever way to do this.
(defun get-params (list)
  "Reading the given list, which is a form of arguments with defun, the function returns a list of symbols"
  (delete-duplicates
   (flatten
    (loop for i fixnum upfrom 0 below (length list)
	  collect (let ((sym (nth i list)))
		    (typecase sym
		      (symbol
		       (if (env-parameter-p sym)
			   nil
			   sym))
		      (list
		       (if (= (length sym) 2)
			   (car sym)
			   (get-params sym)))))))))

(defvar *call-with-view-route* nil)

(defmacro with-tracing-call-with-view (&body body)
  `(let ((*call-with-view-route*)
	 (*ranked-loop-result-cacher*))
     ,@body))

(defun vm-kernel-lambda (traceable? name args self &rest body)
  (make-compiled-kernel
   :name name
   :body body
   :args args
   :self self
   :call-with-view *ranked-loop-result-cacher*
   :cache-when-compiled (if cl-waffe2/vm.generic-tensor::*freeze-call-with-view*
			    nil
			    traceable?)
   :cache-p (when (and traceable? *call-with-view-route*) t)
   :view-route (if (and traceable? *call-with-view-route*)
		   *call-with-view-route*)))

(defmacro with-devices ((&rest backend-priority) &body body)
  "
## [macro] with-devices


The macro `with-devices` declares the priority of dispatching nodes.

### Input

1. `backend-priority` An list of device's name (e.g.: CPUTensor, LispTensor...) Devices on the left have higher priority.

### Example

Let `ATensor` and `BTensor` be a pointer compatible, and subclass of `AbstractTensor`, and operations defined is following:

1. ATensor has !add.
2. BTensor has !mul.

```lisp
(setq a (make-tensor `(10 10))) ;; The tensor a is ATensor.

;; (Priority1=ATensor Priority2=BTensor)
(with-devices (ATensor BTensor)
   (!add a (!mul a a)))
```

cl-waffe2's backend dispatching rule is following:

If the priority 1 backend does not have an implementation of the specified operation, check if the priority 2 backend does, if it still does not have it, 3, 4... and so on.

The order of priority would be `(,@backend-priority ScalarTensor t). (t is a special name, and it implys the implement works for all the backends.)
"
  `(let ((*using-backend* ',backend-priority))
     (declare (type list *using-backend*))
     (mapc
      #'(lambda (x)
	  (unless (subtypep x 'cl-waffe2/vm.generic-tensor:AbstractTensor)
	    (warn "~a is not a subtype of cl-waffe2/vm.generic-tensor:AbstractTensor. If you want to extend device, extend this class first."
		  x)))
      *using-backend*)
     ,@body))

(defmacro with-single-device ((device-name) &body body)
  "Under this macro, cl-waffe only use devices with device-name, otherwise an error will occur"
  `(let ((*facet-monopoly-mode* t))
     (with-devices (,device-name)
       ,@body)))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out)))))
  (defun subnode-name (abstract-node device)
    (intern (with-output-to-string (out)
	      (princ abstract-node out)
	      (princ '- out)
	      (princ device out))
	    'cl-waffe2/vm.nodes.facets-tmp)))

(defun determine-facet-of-nodes (abstract-name devices &rest inputs)
  "Dispathces one of the implementation of AbstractName reading the given devices and inputs"
  (declare (optimize (speed 3))
           (type list devices inputs)
	   (type symbol abstract-name))

  ;; ScalarTensor and T is forcibly added to the last priority
  ;; Reading the device name from higher to lower
  (loop for device in `(,@devices ScalarTensor t)
	do (let ((node-name (subnode-name abstract-name device)))
	     (when (and
		    (find-class node-name nil)
		    (subtypep node-name abstract-name)
		    (node-compatible-p node-name inputs))
	       (return-from determine-facet-of-nodes node-name))

	     (when *facet-monopoly-mode*
	       (error 'node-not-found :node abstract-name))))

  (error 'node-not-found :node abstract-name))


(defmacro subscript (where &key (fixed nil) (allow-symbol nil) (constructor-args nil))
  (multiple-value-bind (body states uprankable parsed-state) (create-subscript-p `,where :fixed fixed :return-body t :allow-symbol allow-symbol :local-variables (get-params `,constructor-args))
    `(values
      ,body
      ',states
      ',uprankable
      ',parsed-state)))

;; Broadcasting Semantic

;; ~ <- broadcastable axes
;; ~ a b, ~ <- batch part, a b <- kernel part
;; Enhancement: add keyword :no-grad which indicates the node doesn't have a implementation of backward.
;; Enhancement: disassemble-me
(defmacro defnode ((abstract-name
		   (self &rest constructor-arguments)
		    &key
		      (where nil)
		      (out-scalar-p nil)
		      (slots nil)
		      (save-for-backward nil)
		      (backward nil)
		      (extends nil)
		      (documentation ""))
		   &body constructor-body
		   &aux (subscript-p  (gensym))
		     (subscript-p1 (gensym))
		     (constructor-arguments `(,self ,@constructor-arguments)))
  "
## [macro] defnode

```lisp
(defnode (abstract-name
		   (self &rest constructor-arguments)
		    &key
		      (where t)
		      (out-scalar-p nil)
		      (slots nil)
		      (save-for-backward nil)
		      (backward nil)
		      (extends nil)
		      (documentation \"\"))
		   &body constructor-body)
```

Declares a new `AbstractNode`.

### Effects
   - defines a class (subclass of `AbstractNode`) named `abstract-name`
   - defines a fucntion which initializes the defined node.

### Inputs

- `abstract-name`[symbol] indicates the name of class, and constructor.
- extends[list] set a list of symbols, the class is defined with extending them.

- `(self &rest constructor-arguments)` declares the arguments of the constructor function, which `cosntructor-body` uses. 

- slots[list] Describe the slots which node has as if defclass. Tips: In order to make it shorter to create a constructor, if initargs (i.e.: `:initarg :XXX`) is the same as the keyword name of the argument, the initform is replaced with the argument.

- where[SubscriptDSL] Put here the Subscript DSL (MUST)

- out-scalar-p [Boolean] Set t if the node returns a ScalarTensor.

- backward [list] This form is optional. The backward receives arguments like: `(dout var1 var2...)` and return tensors which is lazy-evaluated. (See examples). You can set this form as nil, but in that case each `define-impl` and `define-impl-op` must have a backward slot.

- documentation [String]

### Example

```lisp
;; Tips
(defnode (ExampleNode (self arg)
            :slots ((arg :initarg :arg))))

(slot-value (ExampleNode 10) 'arg) ;; => 10

(defnode (MatMulNode-Revisit (myself dtype &key transpose-a transpose-b)
	  :where (A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :slots ((transpose-a :initarg :transpose-a :type boolean :reader trans-a?)
		  (transpose-b :initarg :transpose-b :type boolean :reader trans-b?))
	  :backward ((self dout da db do)
                     ;; dout=previous gradient, :save-for-backward is set to (t t nil).
                     ;; so da/db is a copy of variable.
		     (declare (ignore do))
                     ;; Set nil to the direction gradients aren't produced.
		     (values
		      (!matmul dout (!t db))
		      (!matmul (!t da) dout)
		      nil))
	  :documentation \"
```math
C\\gets{gemm(1.0, A, B, 0.0, C)}
```\"))

You can invoke the forward/backward by using the method forward/backward. `(forward node arg1 arg2...)` `(backward node dout1 arg1 arg2...)`.
"

  (when (null where)
    (error "defnode: Subscript DSL is missing from declaration.

    (defnode (~a (self ...)
               :where ...
                        L___  Fill this form!
               ...)"       
	   abstract-name))
  
  (let ((initarg-slots (map 'list #'(lambda (slots)
				      ;; Auto-Generated Constructor is Enabled Only When:
				      ;; slot has :initarg
				      ;; slot-name corresponds with any of constructor-arguments
				      (when
					  (and
					   (find (first slots) (flatten constructor-arguments))
					   (find :initarg slots))
					slots))
			    slots)))
    ;; where-static-p -> T,   the phase where isn't used
    ;; where-static-p -> NIL, the phase where is used
    
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (defclass ,abstract-name (AbstractNode ,@extends)
	 (,@slots
	  (save-for-backward-space1 :initform ',save-for-backward :reader node-save-for-backward1)
	  (out-scalar-p :initform ,out-scalar-p :accessor out-scalar-p))
	 (:documentation
	  ,(format nil
		   "~%## [node] ~a

```
~a
```

### Description

~a

### Backward

~a"
		   (symbol-name abstract-name)
		   where
		   documentation
		   (if backward
		       (format nil "✅ Already defined. ~%~%```lisp~%~(~a~)~%```~%~%No need to implement backwards at `define-impl`. (they'd be ignored.)" backward)
		       
		       "❌ Undefined. (To make it differentiable, must be defined with `define-impl` macro.)"))))
       ,(when backward
	  (let ((backward-self-name (caar backward))
		(backward-args (cdar backward))
		(backward-body (cdr backward))
		(inputs (gensym "inputs"))
		(impl-name abstract-name))
	    `(defmethod backward ((,backward-self-name ,impl-name) &rest ,inputs)
	       (declare (type ,impl-name ,backward-self-name))
	       (multiple-value-bind (,@backward-args) (apply #'values ,inputs)
		 (declare (type cl-waffe2/vm.generic-tensor:AbstractTensor ,@backward-args))
		,@backward-body))))
       (defmethod print-object ((object ,abstract-name) stream)
	 (format stream
		 "<Node: ~a ~a>"
		 (class-name (class-of object))
		 ',where))
       ;; Backends are modular
       (defun ,abstract-name (,@(cdr constructor-arguments))
	 ,documentation
	 (let* ((,subscript-p  (multiple-value-list (subscript ,where :constructor-args ,(cdr constructor-arguments))))
		(,subscript-p1 (multiple-value-list (subscript ,where :fixed t :constructor-args ,(cdr constructor-arguments)))) ;; subscript-p without ~
		(,(car constructor-arguments)
		  (make-instance
		   (determine-facet-of-nodes ',abstract-name *using-backend* ,@(get-params (cdr constructor-arguments)))
		   :function-node  (car ,subscript-p)
		   :function-node1 (car ,subscript-p1)
		   :uprank-state   (third ,subscript-p)
		   :transmission-state (second ,subscript-p) ;; (second subscript-p) == (second subscript-p1)
		   ,@(loop for slot in initarg-slots
			   if slot
			     collect (intern (symbol-name (nth (1+ (position :initarg slot)) slot)) "KEYWORD")
			   if slot
			     collect (car slot)))))
	   (declare (ignorable ,(car constructor-arguments)))
	   ,@constructor-body
	   (the ,abstract-name ,(car constructor-arguments)))))))

;; TODO: Add Keyword:
;; To Fix: :device=t, -> undefined-byte cl-waffe2/vm.nodes.facets-tmp::...
(defmacro define-impl ((abstract-name
			&key
			  (device t)
			  (extends nil)
			  (cache-when-compiled t)
			  (reject-p nil))
		       &key
			 save-for-backward
			 forward
			 backward
		       &aux
			 (inputs (gensym "inputs")))
  "
## [macro] define-impl

```lisp
(define-impl (abstract-name &key (device t) (extends nil) (cache-when-compiled t) (reject-p nil))
        &key (save-for-backward nil) (forward nil) (backward nil))
```

Defines an implementation of `abstract-name` which is already declared by `defnode` macro, with :forward=macro and later compiled.

### Effects

Defines a CLOS class named `abstract-name-device` extends `abstract-name`

### Inputs

`device`[symbol or t] Set the name of AbstractTensor which the impl supports for. Set t to anything.

`extends`[nil or list] In addition to extend `abstract-name`, the defined implementation will extends the given classses.

`cache-when-compiled`[boolean] Set T to cache the forward definiton depending on dtypes, ranks, devices of arguments. You can set this to NIL but in terms of performance it is not recommended (runtime-compiling overhead is unignorable!) Instead, in that case, using `define-impl-op` would be nice.

`save-for-backward`[list of boolean] For backward computation, the corresponding position of received variables will be produce a copy. You can check how it works with `(disassemble-waffe2-ir toplevel)` function and SV4BW(...) is exactly this. In backward, `((self dout x y) ...)` will receive the copy.

`forward`[body] Follows this format: `((self arg1 arg2 ...) <<macro-body>>)` and the form must return S-expression later compiled by `(compile nil ...)

`backward`[body] Follows this format: `((self prev-gradient arg1 arg2 ...) (values arg1.grad arg2.grad))` Note that the form is given by a function, and computation nodes are continuous. Not a macro.

`reject-p`[nil or function] Set a lambda function returning nil or T. The function is called with arguments: `(function constructor-args1 constructor-args2 ...)`. In the case the function returned T, the method dispatching is ignored. You can use this method to ignore a certain dtype as a :forward arguments for example.
"
  
  (let* ((forward-self-name (caar forward))
	 (backward-self-name (caar backward))
	 (forward-args  (cdar forward))
	 (backward-args (cdar backward))
	 (forward-body  (multiple-value-list (parse-body (cdr forward))))
	 (backward-body (cdr backward))
	 (impl-name (subnode-name abstract-name device))
	 (fw-name-expand (symb abstract-name device '-expand))
	 (fw-name-vm     (symb abstract-name device '-vm-function)))
    
    (eval-when (:compile-toplevel :load-toplevel :execute)
      ;; [TODO] If not deleting all caches, find the impl and remove it.
      (cl-waffe2/vm.generic-tensor:reset-compiled-function-cache!)
      (assert (or (null backward) (= (1- (length backward-args)) (length forward-args)))
	      nil
	      "define-op: The number of arguments do not match: ~a At ~a.
    :forward  should be -> (self arg1 arg2 ...)
    :backward should be -> (self prev-grad arg1 arg2 ...)"
	      backward-args
	      abstract-name))
    
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (progn
	 (assert (or (eql ',device t) (subtypep ',device 'cl-waffe2/vm.generic-tensor:AbstractTensor))
		 nil
		 "define-impl: Assetion Failed because the node ~a 's :device (~a) is not subtype of cl-waffe2/vm.generic-tensor:AbstractTensor."
		 ',abstract-name
		 ',device))
       
       (set-node-reject-case ',impl-name (the (or null function) ,reject-p))
       
       (defclass ,impl-name (,abstract-name ,@extends)
	 ((save-for-backward-space2 :initform ',save-for-backward :reader node-save-for-backward2))
	 (:documentation ,(format nil "The node ~a is a one facet of ~a for the device ~a. Automatically defined by cl-waffe."
				  impl-name
				  abstract-name
				  device)))
       
       ;; TODO: Auto generate of documentations
       (defmethod forward ((,forward-self-name ,impl-name) &rest ,inputs)
	 (declare (type ,impl-name ,forward-self-name))

	 ;; Enhancement: macroexpand
	 (labels ((,fw-name-expand (,inputs)
		    (multiple-value-bind (,@forward-args) (apply #'values ,inputs)
		      ,@(second forward-body)
		      (with-tracing-call-with-view
			(vm-kernel-lambda
			 ,cache-when-compiled ',fw-name-vm ,inputs ,forward-self-name
			 `(named-lambda ,',fw-name-vm ,(map 'list #'tensor-id ,inputs)
			    (declare (ignorable ,@(map 'list #'tensor-id ,inputs)))
			    ,,@(car forward-body)))))))
	   ;; (,fw-name ,inputs) => Expanded Forms.

	   ;; Forms: Lambda (args) -> outs
	   (,fw-name-expand ,inputs)))
       
       
       ;; Backward should be defined at either/both of defnode or/and define-impl. (defnode takes the precendence)
       ,(when backward
	  `(defmethod backward ((,backward-self-name ,impl-name) &rest ,inputs)
	     (declare (type ,impl-name ,backward-self-name))
	     (multiple-value-bind (,@backward-args) (apply #'values ,inputs)
		(declare (type cl-waffe2/vm.generic-tensor:AbstractTensor ,@backward-args))
		,@backward-body))))))

(defun node-save-for-backward (node)
  (or (node-save-for-backward1 node)
      (node-save-for-backward2 node)))


(defmacro define-impl-op ((abstract-name
			   &key
			     (device t)
			     (extends nil)
			     (reject-p nil))
			  &key
			    forward
			    backward
			  &aux
			    (inputs (gensym "inputs")))
  "
## [macro] define-impl-op

Gives an implementation of `abstract-name` as a function form.

```lisp
(define-impl-op ((abstract-name &key (device t) (extends nil) (reject-p nil)) &key forward backward))
```
"
  
  (let* ((forward-self-name  (caar forward))
	 (backward-self-name (caar backward))
	 
	 (forward-args  (cdar forward))
	 (backward-args (cdar backward))
	 
	 (forward-body  (cdr forward))
	 (backward-body (cdr backward))
	 
	 (impl-name     (subnode-name abstract-name device))
	 
	 (fw-name-vm    (symb abstract-name device '-vm-function)))
    
    (eval-when (:compile-toplevel :load-toplevel :execute)
      ;; [TODO] If not deleting all caches, find the impl and remove it.
      (cl-waffe2/vm.generic-tensor:reset-compiled-function-cache!)
      (assert (or (null backward) (= (1- (length backward-args)) (length forward-args)))
	      nil
	      "define-impl-op: The number of arguments in forward and backward should be corresponds. ~a and ~a"
	      backward-args
	      abstract-name))
    
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (progn
	 (assert (or (eql ',device t) (subtypep ',device 'cl-waffe2/vm.generic-tensor:AbstractTensor))
	     nil
	     "define-impl-op: the node ~a 's :device (~a) is not subtype of cl-waffe2/vm.generic-tensor:AbstractTensor."
	     ',abstract-name
	     ',device))
       
       (set-node-reject-case ',impl-name (the (or null function) ,reject-p))
       
       (defclass ,impl-name (,abstract-name ,@extends)
	 ((save-for-backward-space2 :initform nil :reader node-save-for-backward2))
	 (:documentation ,(format nil "The node ~a is a one facet of ~a for the device ~a. Automatically defined by cl-waffe."
				  impl-name
				  abstract-name
				  device)))

       
       (defmethod forward ((,forward-self-name ,impl-name) &rest ,inputs)
	 (declare (type ,impl-name ,forward-self-name))
	 
	 (make-compiled-kernel
	  :op #'(lambda (,@forward-args)
		  (declare (ignorable ,@forward-args))
		  ,@forward-body)
	  :name ',fw-name-vm
	  :body nil
	  :cache-when-compiled nil
	  :cache-p nil
	  :call-with-view nil
	  :args ,inputs
	  :view-route nil
	  :self ,forward-self-name))       
       
       ,(when backward
	  `(defmethod backward ((,backward-self-name ,impl-name) &rest ,inputs)
	     (declare (type ,impl-name ,backward-self-name))
	     (multiple-value-bind (,@backward-args) (apply #'values ,inputs)
		(declare (type cl-waffe2/vm.generic-tensor:AbstractTensor ,@backward-args))
		,@backward-body))))))
	  

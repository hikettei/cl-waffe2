
(in-package :cl-waffe2/vm.nodes)

(defpackage :cl-waffe2/vm.nodes.facets-tmp)

(defparameter *facet-monopoly-mode* nil "This parameter is used to ensure that all the calculations are performed under the same node. If this parameter is t, only use devices with Priority1, otherwise an error will occur.")

(defun movetensor-p (node)
  (subtypep (class-of node) 'cl-waffe2/base-impl:MoveTensorNode))

(defun list-of-abstracttensor-p (list)
  "Return t if LIST is non nil and contains only AbstractTensor."
  (and (consp list)
       (every #'(lambda (x) (subtypep x 'cl-waffe2/vm.generic-tensor:AbstractTensor)) list)))

(deftype list-of-abstracttensor ()
  `(and list (satisfies list-of-abstracttensor-p)))

(defmacro with-devices ((&rest backend-priority) &body body)
  "The macro with-devices declares the node's priority for the function *forward* to be used.

Input:
   - backend-priority
     An list of device's name (e.g.: CPUTensor, LispTensor...)
     Devices on the left have higher priority.

Example:

Let ATensor and BTensor be compatible (i.e.: pointers are the same type), and subclass of AbstractNode, and all the operations they have are as follows:

1. ATensor has !add.
2. BTensor has !mul.

This code works:

(setq a (make-tensor `(10 10))) ;; The tensor a is ATensor.

;; (Priority1=ATensor Priority2=BTensor)
(with-devices (ATensor BTensor)
   (!add a (!mul a a)))

ATensor doesn't have any implementation of !mul, but it does work. This is because cl-waffe2's compatible backend system.

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
	    'cl-waffe2/vm.nodes.facets-tmp))
  (defun tmp-gensym ()
    (intern (symbol-name (symb '* (gensym "C") '*)) 'cl-waffe2/vm.nodes.facets-tmp)))

(defun determine-facet-of-nodes (abstract-name devices)
  (declare (type list devices)
	   (type symbol abstract-name))
  ;; ScalarTensor is forced to use.
  (loop for device in `(,@devices ScalarTensor t)
	do (let ((node-name (subnode-name abstract-name device)))
	     (when (subtypep node-name abstract-name)
	       (return-from determine-facet-of-nodes
		 node-name))

	     (when *facet-monopoly-mode*
	       (error 'node-not-found
		      :node abstract-name))))

  (error 'node-not-found :node abstract-name))


(defmacro subscript (where &key (fixed nil))
  (multiple-value-bind (body states) (create-subscript-p `,where :fixed fixed :return-body t)
    `(values
      (eval ,body)
      ',states)))

(defmacro defnode ((abstract-name
		   (self &rest constructor-arguments)
		    &key
		      (where t)
		      (out-scalar-p nil)
		      (slots nil)
		      (backward nil)
		      (documentation ""))
		   &body constructor-body
		   &aux (subscript-p  (gensym))
		     (subscript-p1 (gensym))
		     (constructor-arguments `(,self ,@constructor-arguments)))
  "The macro defnode declares a generic definition of computation node named abstract-name.

In general, a computation node in cl-waffe2 is defined as:
   1. An data structure with both of forward and backward propagation.
   2. The shape of the matrix before and after the calculation is declared with :where option.
   3. Have a per-device implementation of forward propagation (sometimes backward)

Effects:
   - defines a class (extended AbstractNode) named abstract-name
   - defines a constructor named abstract-name

Inputs:
   - abstract-name[symbol]
   - (self &rest constructor-arguments)
     The constructor will be defined as:
     (defun ,abstract-name (self &rest constructor-arguments)
           ,@construtor-body
           self)
   - slots[list] Describe here the slots node has. (the syntax is the same as defclass's slot.)
   - where[list] Describe pointer movement and shape changes before and after an operation using a DSL with a special syntax.

   - out-scalar-p [Boolean] Set t if the node returns a single ScalarTensor.
   - backward [list, optional] declares the definition of backward. (See also: define-impl)

   - documentation [String]
   - constructor-body [body] describe here the behaviour of constructor.

## How to use where phase?

Place here a small DSL that describes the state of the tensor before and after the operation.

The basic format is that:

```
NAME1[subscripts] NAME2[subscripts] ... -> NAME3[subscripts] NAME4[subscripts] ... where NAMEX = VALUE NAMEY = VALUE
```

It can be divided into three main parts.

[Input-State] -> [Output-State] where [let-binding]

(where [let-binding] can be omitted)

The purpose of this DSL is to represent the shape of the tensor before and after the operation, using undefined symbols, and to identify them automatically.

For Example, When writing a computation node representing the transpose of a two-dimensional matrix, :where phase can be:

```
A[a b] -> A[b a] ;; Corresponding with: (setq x (!transpose x))
```

The function !transpose receives x as a input, and Let X be 10*15 Tensor.

The procedure is that:

1. Refering Input-State, DSL determines all symbols (i.e.: a and b). (If you want to use arbitrary values, define them with let-binding.)

2. Refering determined-symbols, calculates the shape of output-phase.

3. Determine the pointer to use for the output tensor from the described pointer flow.
   In this example, the flow is described as A -> A.

Note that:
  1. The flow of pointers are optional. (i.e.: [~] -> [~] is ok). However, View is no longer recalculated, which can cause bugs, so it is basically better to write it.

  2. using list as let-binding (e.g.: where a = `(1 2 3)) is ok.
  3. ~ is a special symbol, which means that there can be any number of dimensions in between, and the meaning of ~ will be the same in all subscripts. ~ could be determined as nil or list.

### Example1

The Operation is:

```
OUT <- OUT + A
```

In that case, write :where phase like:

```
(defnode (... :where `(OUT[a b] A[a b] -> OUT[a b]) ...))
```

Symbols like OUT, A, indicates what pointer the argument have.

## Tips

In order to simplify parameter initialisation, if the keyword name of the :initarg is the same as the keyword name of the argument, the initialisation code is automatically generated.

(defnode (ExampleNode (self arg)
            :slots ((arg :initarg :arg))))

(slot-value (ExampleNode 10) 'arg) ;; => 10

## Where to place backward?

1.
=================================================================
AddNode (defnode) <- Place Backward
   |
   |-> (AddNode :CPUTensor)  (define-impl)
   |-> (AddNode :LispTensor) (define-impl)
   |-> (AddNode :CUDATensor) (define-impl)
=================================================================

2.
=================================================================
AddNode (defnode) <- Backward=nil
   |
   |-> (AddNode :CPUTensor)  (define-impl) <- place backward
   |-> (AddNode :LispTensor) (define-impl) <- place backward
   |-> (AddNode :CUDATensor) (define-impl) <- place backward
=================================================================

The definition of backward must be placed either of defnode or define-impl.
Basically, if the defnode describes the backward, define-impl's backward is ignored.

Depending on *using-backend*, the implementation to use is determined at node-building time. See also: with-devices.
"
  
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
       (defclass ,abstract-name (AbstractNode)
	 (,@slots
	  (out-scalar-p :initform ,out-scalar-p :accessor out-scalar-p))
	 (:documentation ,documentation))
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
	 (let* ((,subscript-p  (multiple-value-list (subscript ,where)))
		(,subscript-p1 (multiple-value-list (subscript ,where :fixed t))) ;; subscript-p without ~
		(,(car constructor-arguments)
		  (make-instance
		   (determine-facet-of-nodes ',abstract-name *using-backend*)
		   :function-node  (car ,subscript-p)
		   :function-node1 (car ,subscript-p1)
		   :transmission-state (second ,subscript-p) ;; (second subscript-p) == (second subscript-p1)
		   ,@(loop for slot in initarg-slots
			   if slot
			     collect (intern (symbol-name (nth (1+ (position :initarg slot)) slot)) "KEYWORD")
			   if slot
			     collect (car slot)))))
	   (declare (ignorable ,(car constructor-arguments)))
	   ,@constructor-body
	   (the ,abstract-name ,(car constructor-arguments)))))))

;; TODO: Docs
(defmacro define-impl ((abstract-name
			&key
			  (device t))
		       &key
			 save-for-backward
			 forward
			 backward
		       &aux
			 (inputs (gensym "inputs")))
  "Through the macro define-impl, the behaviour of nodes are described.

Follow these constraints:

Memo: forward must return: a list (to be jit-compiled)
      backward must return a values of computation nodes

1. Arguments must be this format:
   Forward  -> (node input-tensor1 input-tensor2 ...)
   Backward -> (node dout dx dy)

   Other parameters should be given as constructor.

Note:
When device=t
save-for-backward's behaviour
"
  
  (let ((forward-self-name (caar forward))
	(backward-self-name (caar backward))
	(forward-args  (cdar forward))
	(backward-args (cdar backward))
	(forward-body  (cdr forward))
	(backward-body (cdr backward))
	(impl-name (subnode-name abstract-name device)))

    (eval-when (:compile-toplevel :load-toplevel :execute)
      (assert (or (null backward) (= (1- (length backward-args)) (length forward-args)))
	      nil
	      "Assertion Failed because the number of arguments of forward and backward, doesn't correspond.: ~a At ~a"
	      backward-args
	      abstract-name))
    
    `(progn
       (eval-when (:compile-toplevel :load-toplevel :execute)
	 (assert (or (eql ',device t) (subtypep ',device 'cl-waffe2/vm.generic-tensor:AbstractTensor))
		 nil
		 "Assetion Failed because the node ~a 's :device (~a) is not subtype of cl-waffe2/vm.generic-tensor:AbstractTensor."
		 ',abstract-name
		 ',device))
       (defclass ,impl-name (,abstract-name)
	 nil
	 (:documentation ,(format nil "The node ~a is a one facet of ~a for the device ~a. Automatically defined by cl-waffe."
				  impl-name
				  abstract-name
				  device)))
       ;; TODO: Auto generate of documentations
       (defmethod forward ((,forward-self-name ,impl-name) &rest ,inputs)
	 (declare (type ,impl-name ,forward-self-name))
	 ;; Make a copy of tensors by coercing them into invoke MoveTensorNode
	 ;; The copy is only needed when cl-waffe2 is used as deep learning framework. (i.e.: backward is called)
	 (loop for input in ,inputs
	       for state in ',save-for-backward
	       if (and state
		       *no-grad*
		       (cl-waffe2/vm.generic-tensor:ancestor-param-p input))
		 do (let ((prev (tensor-backward input)))
		      (if (movetensor-p prev)
			  (setf (cl-waffe2/base-impl:movetensor-save-for-backward prev) t))))
	 
	 (multiple-value-bind (,@forward-args) (apply #'values ,inputs)
	   (declare (type cl-waffe2/vm.generic-tensor:AbstractTensor ,@forward-args))
	   ,@forward-body))

       ;; Backward should be defined at either/both of defnode or/and define-impl. (defnode takes the precendence)
       ,(when backward
	  `(defmethod backward ((,backward-self-name ,impl-name) &rest ,inputs)
	     (declare (type ,impl-name ,backward-self-name))
	     (multiple-value-bind (,@backward-args) (apply #'values ,inputs)
	       (declare (type cl-waffe2/vm.generic-tensor:AbstractTensor ,@backward-args))
	       ,@backward-body))))))


;; TODO: with-embedding-lisp as a alias of this macro.

(defnode (InstantKernelNode (myself call-form)
	  :slots ((call-form :initarg :call-form :type function :reader instant-call-form))
	  :where (A[~] -> A[~])
	  :documentation ""))

(define-impl (InstantKernelNode :device t)
	     :forward ((self x)
		       (let ((result (funcall (instant-call-form self))))
			 (typecase result
			   (list result)
			   (T `(,x)))))
	     :backward ((self dout dx)
			(declare (ignore dx))
			(values dout)))

(defmacro with-instant-kernel (tensor &body body)
  "Creates an instant-kernel following tensor.

This macro is used to embed condition-free Lisp code either in the process of creating a node or after it has been compiled.

Use case:

1. Embedding Lisp Code for building-time.

(setq a (randn `(10 10)))
(with-instant-kernel a
    (print a)) ;; -> (print a) is evaluated

2. Embedding Lisp Code for compile-time.

(setq a (randn `(10 10)))
(with-instant-kernel a
    `(print ,a)) ;; -> (print a) isn't evaluated

(funcall (build *)) ;; -> (print a) will be evaluated.

Note that (equal (with-instant-kernel a) a) is NIL, that is, the returned value of this macro must be followed by a calculation node.

If the return value of Body can be expanded as a macro, the values are compiled together at JIT compile time. Otherwise, the given tensor is returned as is.

TODO: More simple description.
"
  (let ((kernel-name (gensym "InstantKernel")))
    `(flet ((,kernel-name ()
	      ,@body))
       (forward (InstantKernelNode #',kernel-name) ,tensor))))


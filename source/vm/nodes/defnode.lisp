
(in-package :cl-waffe2/vm.nodes)

(defparameter *facet-monopoly-mode* nil "This parameter is used to ensure that all the calculations are performed under the same node. If this parameter is t, only use devices with Priority1, otherwise an error will occur.")

(defparameter *node-reject-case-table* (make-hash-table))

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

(defun list-of-abstracttensor-p (list)
  "Return t if LIST is non nil and contains only AbstractTensor."
  (and (consp list)
       (every #'(lambda (x) (subtypep x 'cl-waffe2/vm.generic-tensor:AbstractTensor)) list)))

(deftype list-of-abstracttensor ()
  `(and list (satisfies list-of-abstracttensor-p)))

;; Is it ok?
(defun env-parameter-p (sym)
  (equal (aref (symbol-name sym) 0) #\&))

;; FixME there must be much clever way to do this.
(defun get-params (list)
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
	    'cl-waffe2/vm.nodes.facets-tmp)))

(defun determine-facet-of-nodes (abstract-name devices &rest inputs)
  (declare (type list devices inputs)
	   (type symbol abstract-name))
  ;; ScalarTensor is forced to use.
  (loop for device in `(,@devices ScalarTensor t)
	do (let ((node-name (subnode-name abstract-name device)))
	     (when (and
		    (node-compatible-p node-name inputs)
		    (subtypep node-name abstract-name))
	       (return-from determine-facet-of-nodes
		 node-name))

	     (when *facet-monopoly-mode*
	       (error 'node-not-found
		      :node abstract-name))))

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
		   :uprank-state (third ,subscript-p)
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
;; reject-when
;; reject-dtype (:uint8 etc...)
(defmacro define-impl ((abstract-name
			&key
			  (device t)
			  (reject-p nil))
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

reject-p = #'(lambda (&rest inputs) ~)
Return t   -> reject
Return nil -> ok
"
  
  (let* ((forward-self-name (caar forward))
	 (backward-self-name (caar backward))
	 (forward-args  (cdar forward))
	 (backward-args (cdar backward))
	 (forward-body  (multiple-value-list (parse-body (cdr forward))))
	 (backward-body (cdr backward))
	 (impl-name (subnode-name abstract-name device))
	 (fw-name (gensym "FW")))
    
    (eval-when (:compile-toplevel :load-toplevel :execute)
      (assert (or (null backward) (= (1- (length backward-args)) (length forward-args)))
	      nil
	      "Assertion Failed because the number of arguments of forward and backward, doesn't correspond.: ~a At ~a"
	      backward-args
	      abstract-name))
    
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (progn
	 (assert (or (eql ',device t) (subtypep ',device 'cl-waffe2/vm.generic-tensor:AbstractTensor))
		 nil
		 "Assetion Failed because the node ~a 's :device (~a) is not subtype of cl-waffe2/vm.generic-tensor:AbstractTensor."
		 ',abstract-name
		 ',device))
       
       (set-node-reject-case ',impl-name (the (or null function) ,reject-p))
       
       (defclass ,impl-name (,abstract-name)
	 nil
	 (:documentation ,(format nil "The node ~a is a one facet of ~a for the device ~a. Automatically defined by cl-waffe."
				  impl-name
				  abstract-name
				  device)))
       
       ;; TODO: Auto generate of documentations
       (defmethod forward ((,forward-self-name ,impl-name) &rest ,inputs)
	 (declare (type ,impl-name ,forward-self-name))

	 ;; Enhancement: macroexpand
	 (flet ((,fw-name (,inputs)
		  (multiple-value-bind (,@forward-args) (apply #'values ,inputs)
		    ,@(second forward-body)
		    `(lambda ,,inputs
		       ,@(loop for input in ,inputs
			       for state in ',save-for-backward
			       if (and state
				       (cl-waffe2/vm.generic-tensor:ancestor-param-p input))
				 collect `(unless *no-grad*
					    (set-save-for-backward ,(tensor-id input))))
		       ,,@(car forward-body)))))
	   ;; (,fw-name ,inputs) => Expanded Forms.
	   (compile nil
		    (map-tree #'(lambda (obj)
				  (typecase obj
				    (AbstractTensor (tensor-id obj))
				    (T obj)))
			      (,fw-name ,inputs)))))
       

       ;; Backward should be defined at either/both of defnode or/and define-impl. (defnode takes the precendence)
       ,(when backward
	  `(defmethod backward ((,backward-self-name ,impl-name) &rest ,inputs)
	     (declare (type ,impl-name ,backward-self-name))
	     (multiple-value-bind (,@backward-args) (apply #'values (map 'list #'(lambda (x) (or (read-save-for-backward x) x)) ,inputs))
	       (declare (type cl-waffe2/vm.generic-tensor:AbstractTensor ,@backward-args))
	       
	       ;; read-save4backwardもそのまま読み込む？
	       ;; i.e.: まとめてコンパイル？
	       ;; Backwardも即時コンパイルにする
	       ;; (self dy) -> grad1 grad2 grad3...
	       ,@backward-body))))))

(defun declare-local-variables (self &rest tensors)
  ""
  (setf (node-local-variables self) tensors))


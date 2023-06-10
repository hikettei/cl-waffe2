
(in-package :cl-waffe2/vm.nodes)

(defpackage :cl-waffe2/vm.nodes.facets-tmp)

(defparameter *facet-monopoly-mode* nil "If t, only use devices with Priority1, otherwise an error will occur.")

(defun movetensor-p (node)
  (subtypep (class-of node) 'cl-waffe2/base-impl:MoveTensorNode))

(defun list-of-abstracttensor-p (list)
  "Return t if LIST is non nil and contains only AbstractTensor."
  (and (consp list)
       (every #'(lambda (x) (subtypep x 'cl-waffe2/vm.generic-tensor:AbstractTensor)) list)))

(deftype list-of-abstracttensor ()
  `(and list (satisfies list-of-abstracttensor-p)))

(defmacro with-devices ((&rest backend-priority) &body body)
  "Through the macro with-devices, the facet of nodes are declared.
backend-priority is described as: (Priority1 Priority2 ...)"
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

(defmacro defnode ((abstract-name
		   (&rest constructor-arguments)
		    &key
		      (where t)
		      (slots nil)
		      (backward nil)
		      (documentation ""))
		   &body constructor-body
		   &aux (subscript-p  (gensym))
		        (subscript-p1 (gensym)))
  "The macro defnode helps you to describe how nodes are working.

abstract-name... The Common Name for your node.

Common Name
 |-> Sub Node(:backend = :cpu)
  ...

=================================================================
Shaping Format:

Ignore with t.

"
  ;; TODO. Error when (length constructor-arguments) = 0 (no name for node)

  (assert (not (= (length constructor-arguments) 0))
	  nil
	  "Assertion Failed because constructor-arguments must satisfy:
(length constructor-arguments) > 0
because it requires a slot for node itself.")
  
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
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (defclass ,abstract-name (AbstractNode)
	 (,@slots)
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
		 "<~a, :where ~a>"
		 (class-name (class-of object))
		 ',where))
       ;; Backends are modular
       (defun ,abstract-name (,@(cdr constructor-arguments))
	 ,documentation
	 (let* ((,subscript-p  (multiple-value-list (create-subscript-p ,where)))
		(,subscript-p1 (multiple-value-list (create-subscript-p ,where :fixed t))) ;; subscript-p without ~
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
                           a abstracttensor (embedded in the computation node)

      backward must return a values of computation nodes

1. Arguments must be this format:
   Forward  -> (node input-tensor1 input-tensor2 ...)
   Backward -> (node dy)

   Other parameters should be given as constructor.

Note:
When device=t

save-for-backward's behaviour

backward's arguments are:"
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
;; FixME: This violates the coding rule because using defun in defun dynamically (If with-instant-kernel isn't called at toplevel).


(defnode (InstantKernelNode (myself call-form)
	  :slots ((call-form :initarg :call-form :type function :reader instant-call-form))
	  :where `(A[~] -> A[~])
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


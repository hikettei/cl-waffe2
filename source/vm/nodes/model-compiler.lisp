
(in-package :cl-waffe2/vm.nodes)

;;
;; This file provides:
;; Compiler from Composite (i.e.: CLOS classes defined by defmodel) into another forms (e.g.: function defnode)
;;


;; [TODO]
;;- retain_graph option
;;- Theano likeな動作を目指したい
;;-  defmodel-as, AbstractNodeのcache
;;-  Dynamically ShapeとControl Flowでもいいけど、コンパイル速度をもっと高速化してPyTorch-likeに動かす方針も可能っぽい
;;     -> キャッシュできないdefine-implをdefine-impl-opで全て置き換えることでcompile nilのオーバーヘッドが0になる
;;     -> call-with-viewのFunction Version apply-rank-iterみたいなのでランクつき演算
;;     -> defpathでFusionOpをすればSoTAに近い性能目指せるか？
;;     -> AD: (log 1 + x)とかのFusionで数値的安定性の保証
;;     -> define-by-run modeでRNN
;;  memory-pool
;;   IR: Block内部でAllocateしたTensorはmemory-poolを出たときにFreeする

;; TODO: defmodel-as ... :whereにoutのシンボル名指定しないとError
;; (make-input `(A)) A=list Tensorにrankを記録させないとAから以降のShapeを推論できなくない？

(defparameter *model-function-cache-form* (make-hash-table))
  
(defun read-where-args (where)
  "where -> (A B) (C D)"
  (multiple-value-bind (in out fw bw) (parse-subscript where)
    (values
     (or in
	 (loop for f in fw
	       for i upfrom 0
	       collect (nth-subscript i)))
     (or out
	 (loop for b in bw
	       for i upfrom 0
	       collect (nth-subscript i))))))

;; model-function-cache-form
;;                    L___ SoftmaxModel(Dtype) ...
;;                    L___  ...

(declaim (type hash-table *model-function-cache-form*))

(deftype model-asif-options ()
  `(and keyword (member :function :node)))

(defun trace-and-compile-composite (need-backward kernel-size-list named composite composite-input-size argument-names &rest args)

;;  (when (some #'(lambda (x) (some #'symbolp (shape x))) args)
;;    (error "defmodel-as: The function ~(~a~) received a tensor which includes dynamic shape.
;;Note that this function isn't subject to lazy-evaluation, and all arguments need to be evaluated." named))

  (when (some #'(lambda (x) (and (tensor-state x) (eql :maybe-not-computed (cl-waffe2/vm.generic-tensor::state-name x (tensor-state x))))) args)
    (warn "defmodel-as: The function ~(~a~) received a tensor where :vec-state=[maybe-not-computed].
Note that this function isn't subject to lazy-evaluation, and all arguments need to be evaluated." named))
  
  (let ((*no-grad* (not need-backward)))
    ;; *freeze-call-with-view*=t and forcibly set force-order=t
    ;; i.e.: Compiled codes are compatible with ND-array
    (let* ((cl-waffe2/vm.generic-tensor::*freeze-call-with-view* t) ;; Loop Collapse shouldn't be done but instead force force-order=t
	   ;;(tensor-names  (map 'list #'(lambda (x) (intern (symbol-name x) "KEYWORD")) names))
	   (batch-lengths (map 'list
			       #'(lambda (x y)
				   (- (cl-waffe2/vm.generic-tensor:dims x) y))
			       args
			       kernel-size-list))
	   (batch-symbols (loop for i upfrom 0 below (apply #'max (map 'list (compose #'cl-waffe2/vm.generic-tensor:dims) args))
				collect (intern (format nil "rank~a" i))))
	   (trace-tensors (loop for arg in args
				for in-size in composite-input-size
				for name in argument-names
				for batch-size in batch-lengths
				collect (make-input (loop for decl-size in in-size
							  for position upfrom 0
							  if (symbol-eq decl-size '~)
							    append
							    (loop for b upfrom 0 below batch-size
								  collect (nth (+ position b) batch-symbols))
							  else
							    append `(,decl-size))
						    (intern (symbol-name name) "KEYWORD")
						    :scalar-p (scalar-p arg)
						    :dtype    (dtype arg)
						    :order    (order arg))))
	   (trace-tensors (loop for tensor in trace-tensors
				for arg    in args
				do (setf (cl-waffe2/vm.generic-tensor:ancestor-param-p tensor)
					 (cl-waffe2/vm.generic-tensor:ancestor-param-p arg))
				   
				if (and need-backward
					(cl-waffe2/vm.generic-tensor:ancestor-param-p arg))
				  collect (progn
					    (setf (slot-value tensor 'requires-grad) t
						  (slot-value tensor 'cl-waffe2/vm.generic-tensor::grad)
						  (make-input (shape tensor) nil
							      :create-from tensor
							      :scalar-p (scalar-p tensor)
							      :dtype    (dtype tensor)
							      :order    (order tensor)))
					    tensor)
				else
				  collect tensor))
	   (toplevel (apply #'call composite trace-tensors)))

      ;; [FixME] !matmul with transpsosed could be compiled as well?
      ;; [TODO]  -> Envolve transpose-p option into lazy-evaluation
      
      (unless (typep toplevel 'AbstractTensor)
	(error "defmodel-as: Attempted to compile the function ~(~a~) but failed because the composite didn't return any AbstractTensor. butgot: ~a
excepted: AbstractTensor"
	       (or named "lambda")
	       toplevel))
      
      (multiple-value-bind (fwiseq bwiseq leaves dout allocation)
	  (cl-waffe2/vm:compile-forward-and-backward toplevel
						     :need-backward need-backward
						     :fuse-p t
						     :compile-mode :default)
	(declare (ignore dout))
	
	;; Detecting Errors
	(when (some #'(lambda (argument-tensor)
			(null (find (tensor-iid argument-tensor)
				    leaves
				    :key #'tensor-iid)))
		    trace-tensors)
	  (error "defmodel-as: Traced the computation node to compile the function ~(~a~) but failed.

This is because the argument ~a wasn't appeared in leaves, that is, your network isn't continuous. Ensure that all tensors used as arguments are also used to compute a result."
		 named
		 (tensor-name
		  (find t trace-tensors
			:test #'(lambda (y argument-tensor)
				  (declare (ignore y))
				  (null (find (tensor-iid argument-tensor)
					      leaves
					      :key #'tensor-iid)))))))
	
	(let ((input-tensors ;; -> (InputTensor(:X), InputTensor(:Y) ...)
		(loop for input-tensor of-type AbstractTensor in trace-tensors
		      append (loop with subject-name = (tensor-name input-tensor)
			           for leaf-point of-type AbstractTensor in leaves
				   when (eql (tensor-name leaf-point) subject-name)
				     collect leaf-point))))
	  
	  ;; Initializes StateContainer (usually created by forward but it does manually in order to change their contents later.)
	  ;; We use StateContainer just to use (read-result) function. so anything is ok for other parameters.
	  
	  (mapc #'(lambda (argument)
		    (setf (tensor-state argument) (make-statecontainer
						   :forward-out-form (make-compiled-kernel))))
		input-tensors)
	  
	  ;; -> AbstractNodeDefinition?
	  #'(lambda (&rest received-arguments &aux (shapes nil) (alloc-as (make-hash-table)))
	      (declare (optimize (speed 3)))
	      (cl-waffe2/vm.generic-tensor::with-adjustable-symbol-scope
		(cl-waffe2/vm::with-static-allocation (allocation)
		  (apply #'shape-compatible? composite received-arguments)
		  
		  (loop for arg of-type AbstractTensor in received-arguments
			for place                      in input-tensors do
			  ;; Update the argument
			  (cl-waffe2/vm::write-result (list place) (list arg))
			  ;; Update the shape
			  (loop for place-name in (shape place)
				for act-val    in (shape arg) do
				  (setf (gethash place-name alloc-as) act-val)
				  (push (cons place-name act-val) shapes)
				  (cl-waffe2/vm.generic-tensor::register-adjustable-shape place-name act-val)))
		  (cl-waffe2/vm::adjust-allocation! allocation alloc-as)
		  (if need-backward
		      (values
		       (eliminate-undetermined-size
			(cl-waffe2/vm:accept-instructions fwiseq))
		       bwiseq
		       shapes
		       trace-tensors
		       allocation)
		      (eliminate-undetermined-size
		       (cl-waffe2/vm:accept-instructions fwiseq)))))))))))

(defun expand-define->function-form (composite where defun-p named
				     &key (need-backward nil))
  (with-gensyms (dispatching-keys found-function)
    (let* ((cache-key (intern (symbol-name (gensym "CF")) "KEYWORD"))
	   (arguments (read-where-args where))
	   (composite-input-size (where->shapes where))
	   (kernel-size-list (loop for shape in composite-input-size
				   collect (- (length shape) (count '~ shape :test #'symbol-eq))))
	   (body
	     (progn
	       `((declare (type AbstractTensor ,@arguments))
		 (let* ((,dispatching-keys
			  ;; Dispatching compiled methods by, :DTYPE, DEVICE, RANK, REQUIRES_GRAD_P
			  (map 'list #'(lambda (tensor) (list (dtype tensor) (class-of tensor) (length (shape tensor)) (cl-waffe2/vm.generic-tensor:ancestor-param-p tensor))) (list ,@arguments)))
			(,found-function (gethash ,dispatching-keys (gethash ,cache-key *model-function-cache-form*))))

		   (if (functionp ,found-function)
		       (funcall ,found-function ,@arguments)
		       (let ((,found-function (trace-and-compile-composite ,need-backward ',kernel-size-list ',named ,composite ',composite-input-size ',arguments ,@arguments)))
			 ;; cache it
			 (setf (gethash ,dispatching-keys (gethash ,cache-key *model-function-cache-form*)) ,found-function)
			 (funcall ,found-function ,@arguments))))))))
      (setf (gethash cache-key *model-function-cache-form*) (make-hash-table :test #'equal))
      (if defun-p
	  `(progn	     
	     (defun ,named (,@arguments)
	       ,@body))
	  (progn
	    `(lambda (,@arguments)
	       ,@body))))))

;; (defclass AbstractStaticCompositeNode () nil)
(defun expand-define->abstractnode (differentiable-p target-model where named)
  (let* ((composite-name (car target-model))
	 (node-name (symb composite-name '-asnode)))
    (multiple-value-bind (in-names out-names in-states out-states let-bindings) (parse-subscript where)
      (declare (ignore let-bindings out-names in-states out-states))
      (with-gensyms (self dy)
	`(progn
	   (define-op (,node-name (,self ,@in-names)
		       :where ,where
		       :slots ((fw-iseq      :initform nil)
			       (bw-iseq      :initform nil)
			       (variables    :initform nil)
			       (dout         :initform nil)
			       (allocation   :initform nil))
		       :forward ((,self ,@in-names)
				 (cl-waffe2/vm::with-static-allocation ((slot-value ,self 'allocation))
				   (let ((out (cl-waffe2/vm:accept-instructions (slot-value ,self 'fw-iseq))))				   
				     (when (scalar-p out)
				       (setf (out-scalar-p ,self) t))
				     out)))

		       :backward ((,self ,dy)
				  ;; multiply-gradients-static(X, Grad) ... X *= Grad
				  
				  (when (null (slot-value ,self 'bw-iseq))
				    (error "Couldn't step a backpropagation of ~a (defined by the defmodel-as macro) because there's no compiled backward InstructionSeq.
=> (defmodel-as (...) :differentiable t)
                              └── Set :differentiable=t or the forward wasn't called."
					   ',node-name))

				  (if (scalar-p (slot-value ,self 'dout))
				      (setf (tensor-vec (slot-value ,self 'dout))
					    (if (scalar-p ,dy)
						(tensor-vec ,dy)
						(cl-waffe2/vm.generic-tensor::vref ,dy 0)))
				      (setf (tensor-vec (slot-value ,self 'dout)) (tensor-vec ,dy)))
				  
				  ;; Call Backward Iseq
				  (cl-waffe2/vm::with-static-allocation ((slot-value ,self 'allocation))
				    (cl-waffe2/vm:accept-instructions (slot-value ,self 'bw-iseq)))

				  ;; Compose gradients
				  (apply #'values
					 (loop for argument in (slot-value ,self 'variables)
					       if (cl-waffe2/vm.generic-tensor:grad argument)
						 collect (cl-waffe2/vm.generic-tensor:grad argument)
					       else
						 collect nil))))

	     ;; Compile in advance
	     (let* (,@(loop for name in in-names
			    collect `(,name (if (cl-waffe2/vm.generic-tensor:ancestor-param-p ,name)
						(cl-waffe2/vm.generic-tensor:parameter ,name)
						,name)))) ;; [FixME] <- name is detached? for reducint compiling time!
	       (setf (slot-value ,self 'variables) (list ,@in-names))
	       (multiple-value-bind (fw-iseq bw-iseq leaves dout alloc)
		   (cl-waffe2/vm:compile-forward-and-backward
		    (call ,target-model ,@in-names)
		    :need-backward ,differentiable-p
		    :fuse-p t
		    :compile-mode :fastest)
		 (declare (ignore leaves))
		 (setf (slot-value ,self 'fw-iseq) fw-iseq
		       (slot-value ,self 'bw-iseq) bw-iseq
		       (slot-value ,self 'dout)    dout
		       (slot-value ,self 'allocation) alloc))))
	   
	   (defun ,named (,@in-names)
	     (declare (type AbstractTensor ,@in-names))
	     ;; in-names=number -> make-tensor auto?
	     (call (,node-name ,@in-names) ,@in-names)))))))
	       
;; [Exported]
(defmacro defmodel-as (target-model
		       &key
			 (where nil)
			 (asif :function)
			 (named nil)
			 (differentiable nil))
  "
## [macro] defmodel-as

```lisp
(defmodel-as target-model &key (where nil) (asif :function) (named nil) (differentiable nil))
```

Redefines a Composite as a new function or AbstractNode specified in the `:asif` keyword. Further functions or `Differentiable AbstractNode` can be defined based on existing Composites (also called as `model` and defined by `defmodel` macro) which bundles several `AbstractNodes`, as long as `:where` form is fulfilled.

**Note that the expanded form includes eval function! So this macro should be placed in the toplevel!**

### Example

```lisp
(defmodel-as (SoftmaxNode) :named static-softmax :asif :function :where (A[~] -> A[~]))
```

### Inputs

`target-model[Composite]` a form to initialize the composite. ~~This from is executed before running the code, and accordingly static.~~

`where[Subscript DSL or null]` If the model has no `:where` declaration, this macro uses this `:where` form instead. Therefore, as long as `defmodel` provides `:where` declaration, this form should be OK if set as nil.

`named[symbol]` this macro will define a new function after `named`. If set to `nil`, the macro return a lambda function instead of defining it. If you're trying to define a new `AbstractNode`, this option should be fulfilled.

`:asif[keyword]` indicates which form the `target-model` is to be redefined, and could be one of:

```
─────────────────────────────────────────────────────────────────────────────────────
  asif    |   description
─────────────────────────────────────────────────────────────────────────────────────
:function | Defines a function to be executed immediately that does not create a node.
─────────────────────────────────────────────────────────────────────────────────────
:node     | Defines a AbstractNode which needs to be compiled later
─────────────────────────────────────────────────────────────────────────────────────
```

### Effects

If `named` is not `NIL`, this macro defines a new function or AbstractNode after `named`.

### Notes

Depending on the `device` and `dtype` used of arguments, several methods are compiled and dispatched.
"

  (when (not (typep asif 'model-asif-options))
    (error "defmodel-as got unexcepted option for :asif.

(defmodel-as target-model :where ... :asif ~a ...)
                                           └── could be one of :function or :node

Choose :asif option from:
─────────────────────────────────────────────────────────────────────────────────────
  asif    |   description
─────────────────────────────────────────────────────────────────────────────────────
:function | Defines a function to be executed immediately that does not create a node.
─────────────────────────────────────────────────────────────────────────────────────
:node     | Defines a AbstractNode which needs to be compiled later
─────────────────────────────────────────────────────────────────────────────────────
"
	   asif))

  (when (and (eql asif :function) differentiable)
    (error "defmodel-as: The option differentiable=t and asif=:function cannot coexist.

(defmodel-as target-model ... :asif :function :differentiable t)
                                       └── Set :asif=:node to make it differentiable.
"))

  (when (and (eql asif :node)
	     (null named))
    (error "defmodel-as: The keyword named should be specified to define a new AbstractNode.
(defmodel-as target-model ... :asif :node :named nil)
                                                  └── Set a name here to be defined.
"))

  (when (and
	 (not (null named))
	 (not (typep named 'symbol)))
    (error "defmodel-as: the option :named is invaild.

(defmodel-as target-model ... :named ~a)
                                     └── :named could be a symbol, and this macro defines a function after `named`.

"
     named))

  (let* ((composite (eval target-model))
	 (where-decl-to-use (or (read-where composite) where)))

    (when (null where-decl-to-use)
      (error "defmodel-as: Attempted to compile into a ~(~a~) but the composite doesn't provide any available :where declaration.

To do this, :where`` should be placed in either defmodel-as or defmodel macro.

(defmodel-as target-model ... :where ...)
                                 └── Add and Specify this keyword

Or

(defmodel (~a (self ...)
              ...
              :where ... <= Add and Specify this keyword
              ...)
      ....)
"
	     asif
	     (car target-model)))

    ;; Check :where form
    (multiple-value-bind (in-names out-names in-state out-state lets) (parse-subscript where-decl-to-use)
      (declare (ignore lets))

      (let ((in (or (null in-names)
		    (not (= (length in-names) (length in-state)))))
	    (out (or (null out-names)
		     (not (= (length out-names) (length out-state))))))
	(when (or in out)
	  (error "defmodel-as: Every subscript should be given their own names by its :where.

~a
~a"
		 (if in
		     "Position: Before ->"
		     "Position: After  ->")
		 
		 (if (null where)
		     ;; From defmodel
		     (format nil "
    (defmodel (~a (self ...)
                  ...
                  :where ~a
                          └── Specify the name to use. (e.g.: A[i j])
                  ...))"
			     (car target-model)
			     where-decl-to-use)
		     ;; From defmodel-as
		     (format nil "
    (defmodel-as ~a
                  ...
                  :where ~a
                          └── Specify the name to use. (e.g.: A[i j])
                  ...))"
			     target-model
			     where-decl-to-use))))))

    (when (and (read-where composite) where)
      (warn "defmodel-as: As both the composite ~a and defmodel-as form declared :where form, defmodel was used in preference.

(defmodel-as target-model ... :where ...)
                                 └── This option is ignored.

"
	    (car target-model)))

    (case asif
      (:function
       `(eval
	 (expand-define->function-form ',target-model ',where-decl-to-use ,(not (null named)) ',named)))
      (:node
       ;; [TODO]
       `(eval
	 (expand-define->abstractnode
	  ,differentiable ',target-model ',where-decl-to-use ',named))))))

;; Utils for multiplying gradients
(defmodel (Multiply-Gradients (self)
	   :on-call-> ((self x grad)
		       (declare (ignore self))
		       (with-no-grad
			 (cl-waffe2/base-impl::A*=B X grad)))))

(defmodel-as (Multiply-Gradients)
  :where (X[~] Grad[~] -> OUT[~])
  :asif :function
  :named multiply-grads-static)


(in-package :cl-waffe2/vm.nodes)

;;
;; This file provides:
;; Compiler from Composite (i.e.: CLOS classes defined by defmodel) into another forms (e.g.: function defnode)
;;

;; TODO: Under *no-inlining-call-with-view*=t declaration, call-with-view function will return a single function (OK)
;; TODO: Compiled-Kernel <- should also have a :force-use-me option, which forcibly compiles the function if set to t. (OK)
;; TODO: proceed should use defmodel-as like way to call functions
;; TODO: Cache it for the future call of proceed
;; TODO: Export APIs and add to the docs
;; TODO: Remove this: MoveTensorNode(SAVE_FOR_BACKWARD) and then, fuseOps (e.g.: ReLuTensorNode, Log1pNode)
;; TODO: (backward AbstractTensor) ;; -> PyTorch/Chainer Mode (i.e.: Pure define by run Mode with no compiling-time ahead)
;; TODO: Update memory-pool. (add use-state attribute)
;; TODO: Update backward
;; TODO: tensors: tensor-delete, dummy-gc-finalize
;; TODO: Composite-Function ... (!matmul !t !t)
;; TODO: delete define-composite-function
;; TODO: Test defmodel-as, documentations
;; TODO: proceed ... TpSort
;; TODO: defmodel-as ... :whereにoutのシンボル名指定しないとError

;; defmodel-as (for abstractnode mo) 動かす
;; proceed-backwardを十分高速に動かす+memory-leakしない
;; -> RNNCellをIterateする

;; (make-input `(A)) A=list Tensorにrankを記録させないとAから以降のShapeを推論できなくない？

;; Memory-Poolの更新と合わせて：
;; Gradientsの加算方法をどうにかする：
;;  計算ノード内で最初の加算 -> MoveTensorNode使えばおk
;;  それ以降 -> Add
;;  GradientのReset -> パラメーターかカウンターを0に治すだけにする (tensor-grad-ref-count 0) instead of ScalarMul

;; ModelのWhereが読めない問題: Encapsulate_NodeでCompositeを一度Wrapする?

;; そしたらdefmodel-asを動かしてみる
;; self ... cache-when-compile=tでも更新

;; 目標1 : defmodel-asと (backward toplevel) でdefine-by-runっぽく動作 + RNNを動作させる
;; 目標2 : memory-pool ... メモリリーク排除, 使う終わったcacheを使い回す, finalizeとかちゃんとする
;;      PR into (DEVELOP)
;; 目標3 : MoveTensorNode(SAVE_FOR_BACKWARD)を排除 + ADの数値的安定性向上とFuseOps

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

(defparameter *model-function-cache-form* (make-hash-table))
;; model-function-cache-form
;;                    L___ SoftmaxModel(Dtype) ...
;;                    L___  ...

(declaim (type hash-table *model-function-cache-form*))

(deftype model-asif-options ()
  `(and keyword (member :function :node)))

(defun trace-and-compile-composite (need-backward kernel-size-list named composite composite-input-size argument-names &rest args)
  
  (when (some #'(lambda (x) (some #'symbolp (shape x))) args)
    (error "defmodel-as: The function ~(~a~) received a tensor which includes dynamic shape.
Note that this function isn't subject to lazy-evaluation, and all arguments need to be evaluated." named))

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
      
      (multiple-value-bind (fwiseq bwiseq leaves)
	  (cl-waffe2/vm:compile-forward-and-backward toplevel
						     :need-backward need-backward
						     :fuse-p t
						     :compile-mode :default)

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
	  #'(lambda (&rest received-arguments &aux (shapes nil))
	      (declare (optimize (speed 3)))
	      (apply #'shape-compatible? composite received-arguments)
	      
	      (cl-waffe2/vm.generic-tensor::with-adjustable-symbol-scope
		(loop for arg of-type AbstractTensor in received-arguments
		      for place                      in input-tensors do
			;; Update the argument
			(cl-waffe2/vm::write-result place (list arg))
			;; Update the shape
			(loop for place-name in (shape place)
			      for act-val    in (shape arg) do
				(push (cons place-name act-val) shapes)
				(cl-waffe2/vm.generic-tensor::register-adjustable-shape place-name act-val)))
		(if need-backward
		    (values
		     (eliminate-undetermined-size
		      (cl-waffe2/vm:accept-instructions fwiseq))
		     bwiseq
		     shapes
		     trace-tensors)
		    (eliminate-undetermined-size
		     (cl-waffe2/vm:accept-instructions fwiseq))))))))))

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
	       (setf (gethash cache-key *model-function-cache-form*) (make-hash-table :test #'equal))
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
      (if defun-p
	  `(defun ,named (,@arguments)
	     ,@body)
	  `(lambda (,@arguments)
	     ,@body)))))

;; argsのパラメーターで検索かける
;; Backward ...
;; define-static-node ... backward 複数の勾配いけるっけ？
;; trace-and-compileで逆伝播を生成
;; 下を使えばコンパイルをcacheする
;; whereが使えない
;; out-scalar-p
;; outの数=1でないといけないという制約？？
;; outの数=2で微分できますか？
;; lazy-valuesみたいなノードを作る？？？(Systemだけが使う not for users) <- テストもっと増やしたほうがいい

;; (defclass AbstractStaticCompositeNode () nil)

(defun expand-define->abstractnode (differentiable-p target-model where named)
  (let* ((composite-name (car target-model))
	 (node-name (symb composite-name '-asnode)))
    (multiple-value-bind (in-names out-names in-states out-states let-bindings) (parse-subscript where)
      (declare (ignore let-bindings out-names in-states out-states))
      (with-gensyms (self dy kernel-function)
	`(let ((,kernel-function ,(expand-define->function-form
				   target-model
				   where
				   nil
				   nil
				   :need-backward differentiable-p)))
	   ;; ,kernel-function -> result1 result2 ... backward-iseq
	   (define-static-node (,node-name (,self)
				:where ,where
				:slots ((bwiseq      :initform nil)
					(variables   :initform nil)
					(shape-table :initform nil))
				:cache-when-compiled nil ;; TODO set=t to make compiling-overhead zero
				:forward ((,self ,@in-names)
					  (multiple-value-bind (out bwiseq shapes variables) (apply ,kernel-function (list ,@in-names))
					    (setf (slot-value ,self 'bwiseq) bwiseq
						  (slot-value ,self 'shape-table) shapes
						  (slot-value ,self 'variables) variables)
					    (when (scalar-p out)
					      (setf (out-scalar-p ,self) t))
					    out))
				:backward ((,self ,dy)
					   ;; multiply-gradients-static(X, Grad) ... X *= Grad
					   (when (null (slot-value ,self 'bwiseq))
					     (error "Couldn't step a backpropagation of ~a (defined by the defmodel-as macro) because there's no compiled backward InstructionSeq.
=> (defmodel-as (...) :differentiable t)
                              └── Set :differentiable=t or the forward wasn't called."
						    ',node-name))
					   
					   ;; Call backwards
					   (let ((shapes (slot-value ,self 'shape-table)))
					     (cl-waffe2/vm.generic-tensor::with-adjustable-symbol-scope
					       
					       (loop for comb in shapes do
						 (cl-waffe2/vm.generic-tensor::register-adjustable-shape (car comb) (cdr comb)))
					       
					       (cl-waffe2/vm:accept-instructions (slot-value ,self 'bwiseq))))
					   ;; Pass gradients to the next nodes
					   ;; Each gradients are destructed
					   
					   (apply #'values
						  (loop for argument in (slot-value ,self 'variables)
							if (cl-waffe2/vm.generic-tensor:grad argument)
							  collect (multiply-grads-static
								   (cl-waffe2/vm.generic-tensor:grad argument)
								   ,dy)
							else
							  collect nil)))))
	   
	   (defun ,named (,@in-names)
	     (declare (type AbstractTensor ,@in-names))
	     ;; in-names=number -> make-tensor auto?
	     (call (,node-name) ,@in-names)))))))
	       

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

Creates a new function or AbstractNode from `Composites`. Further functions or `Differentiable AbstractNode` can be defined based on existing Composites (also called as `model` and defined by `defmodel` macro) which bundles several `AbstractNodes`, as long as `:where` form is fulfilled.

### Example

```lisp
(defmodel-as (SoftmaxNode) :named static-softmax :asif :function :where (A[~] -> A[~]))
```

### Inputs

`target-model[Composite]` specify a initializing form of `Composite` to be redefined.

`where[Subscript DSL or null]` If the model has no `:where` declaration, this macro uses this `:where` form instead. Therefore, as long as `defmodel` provides `:where` declaration, this form should be OK if set as nil.

`named[symbol]` this macro will define a new function after `named`. If set to `nil`, the macro return a lambda function instead of defining it. If you're trying to define a new `AbstractNode`, this option should be fulfilled.

`:asif[keyword]` indicates which form the `target-model` is to be redefined, and could be one of:

```
─────────────────────────────────────────────────────────────────────────────────────
  asif    |   description
─────────────────────────────────────────────────────────────────────────────────────
:function | Defines a function to be executed immediately that does not create a node.
─────────────────────────────────────────────────────────────────────────────────────
:node     | Defines a AbstractNode which needs to be compiler later
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
:node     | Defines a AbstractNode which needs to be compiler later
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

    (when (and (read-where composite) where)
      (warn "defmodel-as: As both the composite ~a and defmodel-as form declared :where form, defmodel was used in preference.

(defmodel-as target-model ... :where ...)
                                 └── This option is ignored.

"
	    (car target-model)))

    (case asif
      (:function
       (expand-define->function-form
	target-model where-decl-to-use (not (null named)) named))
      (:node
       ;; [TODO]
       (expand-define->abstractnode
	differentiable target-model where-decl-to-use named)))))


(defmodel (Multiply-Gradients (self)
	   :on-call-> ((self x grad)
		       (declare (ignore self))
		       (with-no-grad
			 (cl-waffe2/base-impl:A*=B X grad)))))

(defmodel-as (Multiply-Gradients)
  :where (X[~] Grad[~] -> X[~])
  :asif :function
  :named multiply-grads-static)

(defmodel (SinModel (self) :on-call-> ((self x) (cl-waffe2/base-impl:!sin x))))
(defmodel-as (SinModel) :where (X[~] -> Out[~]) :asif :node :named !sinmodel :differentiable t)

(defmodel-as (Multiply-Gradients)
  :where (X[~] Grad[~] -> out[~])
  :asif :node
  :named !mgrad
  :differentiable t)

;; Backward?
;; -> Test
;; Backward ... In-place のせいで うまく動かない・・・


(in-package :cl-waffe2/vm.generic-tensor)

;;
;; acceptor.lisp provides an compiler of given nodes, and general-purpose APIs to handle with cl-waffe2 nodes.
;;  With regard to cl-waffe2 VM/IR, see :cl-waffe2/vm package.
;;

(defparameter *no-grad* nil "
## [parameter] `*no-grad*`

Ensures that back-propagation is not invoked inside the scope for which this parameter is set to T, with the following effects:

- Save For Backward is forcibly ignored.

- Computational nodes for back propagation are not compiled.

In default, set to nil. See also the `with-no-grad` macro to explict this state.
")

(defmacro with-no-grad (&body body)
  "
## [macro] with-no-grad

```lisp
(with-no-grad &body body)
```

Set T to `*no-grad*` during the execution of body.
"
  `(let ((*no-grad* t))
     ,@body))


;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; The process of compiling from AbstractNode into cl-waffe2 IR
;;
;;  1. [forward] X Y... -> Produce StateContainer
;;  2. [build]          -> Reading the value of StateContainer, cl-waffe2 creates a Compiled-Kernel structure
;;  3. [opt]            -> After sorting/replacing the nodes, cl-waffe2 creates a InstructionsSeq, which is a list of compiled-kenrel.
;;
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(defstruct (StateContainer)
  "
## [struct] StateContainer

The StateContainer structure is a Compiled-Kernel structure that stores before and after the instruction is compiled and the result of the instruction.
"
  (state :initialized :type (member :initialized :forwarded :backwarded))
  (latest-p nil :type boolean) ;; Synchronized with Memory-pool? (if the tensor is InputTensor)
  
  (forward-out-form nil :type Compiled-Kernel) ;; Instruction Info
  (forward-result   nil :type (or null AbstractTensor))

  (backward-input-variable)
  (backward-out-form nil :type list)
  (backward-result   nil :type list)

  (forward-n-out  0 :type fixnum)
  (backward-n-out 0 :type fixnum))

;; TODO: read-result should be inlined
(declaim (inline read-result)
	 (ftype (function (AbstractTensor) AbstractTensor) read-result))
(defun read-result (tensor)
 "Returns the result of computing of tensor in the compiled code"
  (declare (type AbstractTensor tensor))

  (let ((state (tensor-state tensor)))
    (if state
        (statecontainer-forward-result (tensor-state tensor))
	tensor)))

(defstruct (NodeVariables
	    (:constructor make-variable-table
		(parameters
		 symbols
		 adjustable-symbol-table
		 variables)))
  "
## [struct] NodeVariables

The NodeVariables structure stores the tensors and Shapes that have been lazily evaluated in the computation node.
"
  (parameters parameters :type list)
  (symbols       symbols :type list)   ;; 'a -> current-size 'b -> current-size
  (adjustable-symbol adjustable-symbol-table :type hash-table) ;; 'a -> max-alloc-size 'b -> max-alloc-size
  (variables     variables :type hash-table)) ;; (make-input `(...) :A)

(defmethod print-object ((node NodeVariables) stream)
  (let* ((vars  (nodevariables-variables node)) 
	 (input-keys (alexandria:hash-table-keys vars)))
	 
    (format
     stream
     "    inputs:
~a"

     (with-output-to-string (out)
       (let ((first-row)
	     (second-row))
	 (loop for k upfrom 0 below (length input-keys)
	       do (push (nth k input-keys) first-row)
		  (push (shape (gethash (nth k input-keys) vars)) second-row))
	 
	 (mapc #'(lambda (n s)
		   (format out "        ~a -> ~a~%" n s))
	       (reverse first-row) (reverse second-row)))))))

(defun embody-input (nodevars variable-name actual-tensor)
  "
## [function] embody-input

Giving values to delay-evaluated tensors.
"
  (declare (type NodeVariables nodevars))
  (let ((input-tensor (gethash variable-name (nodevariables-variables nodevars))))
    
    (when (null input-tensor)
      (error "set-input: The InputTensor named ~a weren't appeared in the computation node." variable-name))

    (when (not (= (dims input-tensor) (dims actual-tensor)))
      (error "set-input: the place and value should have the same rank: ~a and ~a" input-tensor actual-tensor))
    
    (let ((symbols-changed (make-hash-table)))
      (loop for place in (tensor-input-shape input-tensor)
	    for value in (shape actual-tensor)
	    for rank upfrom 0
	    if (and (not (symbolp place))
		    (not (= place value)))
	      do (error "set-input: Can't set ~a as ~a of the InputTensor ~a because it is a fixnum.
The operation was: Setting ~a <- ~a
"
			place
			value variable-name
			(tensor-input-shape input-tensor)
			(shape actual-tensor))
	    if (symbolp place)
	      do (setf (gethash place symbols-changed) value))
            
      ;; InputTensor <- Actual-Tensor
      (embody-actual-tensor input-tensor actual-tensor)
      
      ;; Update hash-table
      (maphash
       #'(lambda (key value)
	   (setf (getf (nodevariables-symbols nodevars) key) value))
       symbols-changed)
      t)))

(defun construct-variables-table (variables-list)
  "Returns variable-table where key and value is tensor's name and tensor's pointer."
  (declare (type list variables-list))
  (let ((variable-table (make-hash-table))
	(adjustable-symbol-table (make-hash-table))
	(parameters `())
	(symbols `()))

    (mapc
     #'(lambda (tensor)
	 (let ((shapes (shape tensor)))
	   (loop for s in shapes
		 if (symbolp s)
		   do (setf (gethash s adjustable-symbol-table) nil)
		      (setf (getf symbols s) nil)))

	 ;; tensor-name is keyword -> user-defined
	 ;; tensor-name is string  -> automatically-generated (assertion make not for users to never declare it)

	 ;; (make-input ... :Train-X)
	 (when (slot-value tensor 'requires-grad)
	   (push tensor parameters))
	 
	 (when (user-input-p tensor)
	   (setf (gethash (tensor-name tensor) variable-table) tensor)))
     variables-list)

    (make-variable-table
     parameters
     symbols
     adjustable-symbol-table
     variable-table)))

(defun copy-variable-table (table allocation)
  (declare (type NodeVariables table))
  (let ((table (make-variable-table
		(nodevariables-parameters        table)
		(nodevariables-symbols           table)
		(nodevariables-adjustable-symbol table)
		(alexandria:copy-hash-table
		 (nodevariables-variables table)))))
    (flet ((replace-new (tensor)
	     (or (gethash (tensor-id tensor) (cl-waffe2/vm::vmalloc-id2pool allocation))
		 tensor)))
      (maphash
       #'(lambda (place tensor)
	   (setf (gethash place (nodevariables-variables table))
		 (replace-new tensor)))
       (nodevariables-variables table))
      table)))

;; ==========================================
;; General-Purpose APIs
;; ==========================================

(defclass Compiled-Composite ()
  ((compiled-forward :initarg :compiled-forward :type function :reader compiled-forward)
   (compiled-backward :initarg :compiled-backward :type (or null function) :reader compiled-backward)
   (allocation :initarg :allocation :reader compiled-allocation)
   (variables :initarg :variables :reader compiled-variables)
   (inputs :initform nil :initarg :inputs :reader compiled-inputs)
   (out    :initform nil :initarg :out    :reader compiled-out)
   (dout   :initform nil :initarg :dout :reader compiled-dout)
   (first-call-p :initform nil :accessor composite-first-call-p :type boolean))
  (:documentation "
## [class] Compiled-Composite

Stores information on computation nodes compiled by the build function. The user has to guarantee that this point is the end of the computation node. Therefore, it is not possible in principle to continue the computation node after this point. Forward and backward propagation can be invoked using the `forward` and `backward` methods respectively.

```lisp
;; Example
(let ((model (build (!add 1 1))))
      (forward model)
      (backward model))
```

This class furthermore records information on lazy-evaluated tensors. The tensor is an argument to the function, which can change the input via the `set-input` method.

```lisp
(let ((lazy-tensor (make-input `(10 10) :A)))
    (let ((model (build (!sum lazy-tensor))))
         (set-input model :A (randn `(10 10))) ;; :A = (randn `(10 10))
         (get-input model :A)
         (forward model)))
```

By passing argument information to the compiler at build time, arguments can be given together when the forward method is called.

```lisp
(let ((a (make-input `(A B) :A))
      (b (make-input `(A B) :B)))
    (let ((model (build (!mul a b) :inputs `(:A :B))))
          (forward model (randn `(3 3)) (randn `(3 3)))))
```

All tensors with `:requires-grad=t`, can be accessed by the `(model-parameters model)` method.
"))

(defmethod model-parameters ((model Compiled-Composite))
  "
## [function] model-parameters

Returns a list of tensors with (:requires-grad=t)"
  (nodevariables-parameters (compiled-variables model)))

(defun all-embodied? (model)
  "Invokes an simple-error when model has still unembodied tensor"
  (declare (type Compiled-Composite model))
  (let ((vars (nodevariables-variables (compiled-variables model)))
	(unembodied))
    (loop for key being the hash-keys in vars using (hash-value val)
	  if (null (vec val))
	    do (push (cons key val) unembodied))

    (when unembodied
      (error "The compiled model ~a still have an unembodied tensors.
Before calling the forward method, set any value to these InputTensors first.
~a"
	     model
	     (with-output-to-string (out)
	       (dolist (k unembodied) (format out "~% :~a -> ~a" (car k) (shape (cdr k)))))))))

(defun set-adjustable-symbols (model)
  (let* ((var-table (compiled-variables model))
	 (symbols   (nodevariables-symbols var-table))
	 (allocator (make-hash-table))
	 (lazyaxis-list))
    (loop for i fixnum upfrom 0 below (length symbols) by 2 do
      (let ((symbol-place (nth i symbols))
	    (symbol-value (read-symbol (nth (1+ i) symbols)))) ;; If symbol is registed at the prior scope? read fixnum as possible
	(if (cl-waffe2/vm:symbol-lazyaxis symbol-place)
	    ;; LazyAxis -> Determine later
	    (push symbol-place lazyaxis-list)
	    (progn
	      (when (null symbol-value)
		(error "set-adjustable-symbols: the adjustable symbol ~a -> ~a wasn't registerd well?" symbol-place symbol-value))
	      (register-adjustable-shape symbol-place symbol-value)
	      (setf (gethash symbol-place allocator) symbol-value)))))

    ;; Given symbols ignoring LazyAxis, determine the result of LazyAxis
    (dolist (lazyaxis lazyaxis-list)
      (let* ((axis (cl-waffe2/vm:symbol-lazyaxis lazyaxis))
	     (val  (block try-make-it-static
		     (handler-bind
			 ((error #'(lambda (cond)
				     (format t "The axis ~a cannot be inferred as a static shape because of ~a. Proceed with interpreting as a dynamic tensor.~%" axis cond)
				     (return-from try-make-it-static -1))))
		       (cl-waffe2/vm:observe-axis axis)))))
	
	(when (null val)
	  (error "set-adjustable-symbols: the adjustable symbol ~a wasn't registered well?" axis))

	(register-adjustable-shape lazyaxis val)
	(setf (gethash lazyaxis allocator) val)))
    allocator))

(defmethod cl-waffe2/vm.nodes:forward ((model Compiled-Composite) &rest inputs)
  (let ((input-args (the list (compiled-inputs model))))
    (when input-args
      (assert (= (the fixnum (length input-args)) (the fixnum (length inputs)))
	  nil
	  "forward: Can't invoke the forward step of given Compiled-Composite because the number of arguments received is invalid.
    (forward compiled-model~a)
                            └── the model is compiled as~a."
	  (with-output-to-string (out)
	    (dotimes (i (length inputs)) (format out " inputs[~a]" i)))
	  (with-output-to-string (out)
	    (dolist  (i input-args) (format out " ~a" i))))

      (let ((shape-table (make-hash-table))
	    (places (map 'list #'(lambda (n) (get-input model n)) input-args)))
	(loop for val   in inputs
	      for place in places
	      for name  in input-args do
		(loop for shape in (tensor-input-shape place)
		      for value in (shape val)
		      if (or (null (gethash shape shape-table))
			     (= (gethash shape shape-table) (read-symbol value)))
			do (setf (gethash shape shape-table) (read-symbol value))
		      else
			do (error "forward: Can't forward compiled-composite due to shape-error of inputs.
At:
~a

(forward compiled-model~a)
                        └── Expected:~a
                            But got:  ~a"
				  model
				  (with-output-to-string (out)
				    (dolist (arg places) (format out " ~a" (tensor-name arg))))
				  (with-output-to-string (out)
				    (dolist (arg places) (format out " ~a" (tensor-input-shape arg))))
				  (with-output-to-string (out)
				    (dolist (arg inputs)     (format out " ~a" (shape arg)))))
		      finally
			 (when (not (= (length (tensor-input-shape place))
				       (length (shape val))))
			   (error "forward: Can't forward compiled-composite due to rank-error of inputs.
At:
~a

(forward compiled-model~a)
                        └── Expected:~a
                            But got:  ~a"
				  model
				  (with-output-to-string (out)
				    (dolist (arg places) (format out " ~a" (tensor-name arg))))
				  (with-output-to-string (out)
				    (dolist (arg places) (format out " ~a" (tensor-input-shape arg))))
				  (with-output-to-string (out)
				    (dolist (arg inputs)     (format out " ~a" (shape arg)))))))
		(set-input model name val)))))
  
  ;; Check if all the inputs are embodied?
  (let ((*runtime-mode-p* t))
    (funcall (compiled-forward model) model)))

(defmethod cl-waffe2/vm.nodes:backward ((model Compiled-Composite) &rest inputs)
  (when inputs
    (warn "backward:
    (backward compiled-model inputs)
                              └── inputs for compiled-model is ignored"))
  
  (when (null (compiled-backward model))
    (error "backward:
    (backward compiled-model)
                   └── The backward isn't compiled. Perhaps this is because the model is compiled under (with-no-grad ...) macro."))

  (let ((*runtime-mode-p* t))
    (funcall (compiled-backward model) model))
  t)

(defmethod set-input ((model Compiled-Composite) input-name actual-value)
  "
## [method] set-input

```
(set-input (model Compiled-Composite) input-name actual-value)
```

Embodies an `InputTensor` in the model. All unembodied tensors in the model can be accessed by printing the model.

`input-name` could be a keyword indicating input-tensor, `actual-value` is a `AbstractTensor` whose facet = `:exist` (created by `make-tensor`).
"
  (embody-input (compiled-variables model) input-name actual-value))

(defmethod get-input ((model Compiled-Composite) input-name)
  "
## [method] get-input

```
(get-input (model Compiled-Composite) input-name)
```

Reading all variables in the computation node, the method get-input returns an corresponding `InputTensor` of model.
"
  (gethash input-name (nodevariables-variables (compiled-variables model))))

(defun build (toplevel
	      &key
		(inputs nil)
		(construct-backward? (not *no-grad*))
		(compile-mode :default)
		(fuse-ops t)
		(defmodel-as-from nil)
		(dout-add1 t))
  "
## [function] build

```lisp
(build toplevel &key (inputs nil) (construct-backward? (not *no-grad*)) (compile-mode :fastest) (fuse-ops t))
```

Compiles the given computation node starting from `toplevel`. The docstring of `Compiled-Composite` describes how this function are used in practical.

### Inputs

`toplevel [AbstractTensor or List]` The end of the computational graph. If a list of abstracttensors is provided, the function uses `cl-waffe2/base-impl:lazy-values` to merge them into a single tensor and starts the compilation.

`inputs[list]` Set a list of argument keywords here so that the method `forward` can receive arguments that have been lazily evaluated. The order is taken into account. (e.g.: Set to `(:A :B)` and forward can receive this: `(forward compiled-model (randn `(3 3)) (randn `(3 3)))`)

`construct-backward?` [boolean] Set t to build backward.

`compile-mode`[compile-mode-t] an keyword indicating the compiling option. (No significant impact on execution speed but compile speed. for any case `:fastest` is the best solution.)

`fuse-ops[boolean]` Set to enable `FusionOps` declared by `defpath`.
"
  (declare (type (or list AbstractTensor) toplevel))

  (when (listp toplevel)
    (assert (every #'(lambda (x) (typep x 'AbstractTensor)) toplevel)
	    nil
	    "build: `toplevel` expected AbstractTensor, or a list of abstracttensor.
    (build toplevel ...)
             └── butgot: ~a
"
	    toplevel)
    (setf toplevel (apply #'cl-waffe2/base-impl:lazy-values toplevel)))

  (when inputs
    (assert (every #'keywordp inputs)
	nil
	"build: Can't compile the tensor because the :inputs list is malformed or invalid.
    (build toplevel :inputs ( ... ) ...)
                              └── :inputs receive a list of keyword indicating the name of tensors created by make-input
                                  Set like `(:A :B) "))

  (with-empty-cached-function-table
    (multiple-value-bind (fw-iseq bw-iseq variables dout allocation)
	(cl-waffe2/vm:compile-forward-and-backward toplevel
						   :need-backward construct-backward?
						   :compile-mode compile-mode
						   :fuse-p fuse-ops
						   :add1 dout-add1)
      
      (let ((forward-f  #'(lambda (model)
			    (with-adjustable-symbol-scope
			      (let ((alloc-inst (set-adjustable-symbols model)))
				(cl-waffe2/vm::with-static-allocation ((compiled-allocation model))
				  (cl-waffe2/vm::adjust-allocation! (compiled-allocation model) alloc-inst)
				  (all-embodied? model)
				  (cl-waffe2/vm:accept-instructions fw-iseq))))))
	    (backward-f   (when construct-backward?
			    #'(lambda (model)
				(with-adjustable-symbol-scope
				  (let ((alloc-inst (set-adjustable-symbols model)))
				    (cl-waffe2/vm::with-static-allocation ((compiled-allocation model))
				      (cl-waffe2/vm::adjust-allocation! (compiled-allocation model) alloc-inst)
				      (cl-waffe2/vm:accept-instructions bw-iseq)))))))
	    (table        (construct-variables-table variables)))


	;; Check all arguments are valid as an argument
	(when inputs
	  (let ((input-names (alexandria:hash-table-keys (nodevariables-variables table))))
	    (mapc
	     #'(lambda (x)
		 (when (not (find x input-names))
		   (error "~abuild: Can't compile the tensor because the argument ~a didn't appear in the computation node.
        (build toplevel :inputs ~a)
                                └── Choose from: ~a
Or, your network may be disconnected at a certain position."
			  (if defmodel-as-from
			      (format nil "~%defmodel-as: Attempted to compile the function ~(~a~) but failed due to:~%" defmodel-as-from)
			      "")
			  x
			  inputs
			  input-names)))
	     inputs)))

	(make-instance 'Compiled-Composite
		       :allocation allocation
		       :compiled-forward  forward-f
		       :compiled-backward backward-f
		       :out               toplevel
		       :dout              dout
		       :inputs            inputs
		       :variables         table)))))

(defmethod copy-compiled-model ((model Compiled-Composite))
  (let* ((allocation (cl-waffe2/vm::copy-allocate (compiled-allocation model)))
	 (table      (cl-waffe2/vm::vmalloc-id2pool allocation)))
    (flet ((replace-new (tensor)
	     (when tensor
	       (or (gethash (tensor-id tensor) table)
		   tensor))))
      (let ((dout (replace-new (compiled-dout model)))
	    (out  (replace-new (compiled-out  model))))
	(make-instance 'Compiled-Composite
		       ;; Duplicate Memory-Pool
		       :allocation allocation
		       
		       :compiled-forward  (compiled-forward model)
		       :compiled-backward (compiled-backward model)

		       :dout dout
		       :out  out
		       
		       :inputs    (compiled-inputs model) ;; a list of keywords
		       :variables (copy-variable-table (compiled-variables model) allocation))))))

;; TODO -> (defmethod free-model ((model Compiled-Composite)))
;; TODO -> (defmethod model-memory-size ((model Compiled-Composite)) )

(defmethod model-memory-size ((model Compiled-Composite))
  "Return: mempool-size"
  (let ((mempool-fixed-size 0)
	(adj-p nil)
	(adjust-size (make-hash-table :test #'equal)))

    (maphash
     #'(lambda (id tensor)
	 (declare (ignore id))
	 (multiple-value-bind (memsize fixed-area ndm) (tensor-memory-size tensor)
	   (if ndm
	       (progn
		 (setq adj-p t)
		 (incf mempool-fixed-size (* memsize fixed-area))
		 (setf (gethash ndm adjust-size)
		       (if (gethash ndm adjust-size)
			   (+ (gethash ndm adjust-size) memsize)
			   memsize)))
	       (incf mempool-fixed-size memsize))))
     (cl-waffe2/vm::vmalloc-id2pool (compiled-allocation model)))

    (format nil "{~a~a}MB"
	    (/ mempool-fixed-size 1e+6)
	    (if adj-p
		(with-output-to-string (out)
		  (maphash
		   #'(lambda (size stride)
		       (format out "+(~a x ~a)~a"
			       size
			       (/ stride 1e+6)
			       (if (<= (hash-table-count adjust-size) 1)
				   ""
				   (format nil "~%                      "))))
		   adjust-size))
		""))))

(defmethod print-object ((model Compiled-Composite) stream)
  (format stream "<Compiled-Composite(allocated-p=~a)
    forward     : ~a
    backward    : ~a
    memory-pool : ~R tensor(s)
                   L ~a
~a>"
	  ;; Variables
	  (cl-waffe2/vm::vmalloc-allocated-p (compiled-allocation model))
	  (if (compiled-inputs model)
	      (with-output-to-string (out)
		(format out "forward(model")
		(dolist (i (compiled-inputs model)) (format out " ~a" i))
		(format out ") -> ~a{~a}~a" (class-name (class-of (compiled-out model))) (dtype (compiled-out model)) (shape (compiled-out model))))
	      (format nil "forward(model) -> ~a{~a}~a" (class-name (class-of (compiled-out model))) (dtype (compiled-out model)) (shape (compiled-out model))))
	  (if (compiled-backward model)
	      "backward(model) -> t"
	      "nil")
	  (hash-table-count (cl-waffe2/vm::vmalloc-id2pool (compiled-allocation model)))
	  (model-memory-size model)
	  (if (>= (length (alexandria:hash-table-keys (nodevariables-variables (compiled-variables model)))) 1)
	      (compiled-variables model)
	      "")))



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
		 variables
		 tmp-variables)))
  "
## [struct] NodeVariables

The NodeVariables structure stores the tensors and Shapes that have been lazily evaluated in the computation node.
"
  (parameters parameters :type list)
  (symbols       symbols :type list)   ;; 'a -> current-size 'b -> current-size
  (adjustable-symbol adjustable-symbol-table :type hash-table) ;; 'a -> max-alloc-size 'b -> max-alloc-size
  (variables     variables :type hash-table) ;; (make-input `(...) :A)
  (tmp-variables tmp-variables :type list))  ;; (make-input `(...) nil) i.e.: chaintmp

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
	(tmp-variable-table) ;; All the InputTensor
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
	   (setf (gethash (tensor-name tensor) variable-table) tensor))

	 ;; (make-input ... ChainTMPXXX)
	 (when (typep (tensor-name tensor) 'string)
	   (push tensor tmp-variable-table)))
     variables-list)

    (make-variable-table
     parameters
     symbols
     adjustable-symbol-table
     variable-table
     tmp-variable-table)))

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
	 (allocator (make-hash-table)))
    (loop for i fixnum upfrom 0 below (length symbols) by 2
	  do (register-adjustable-shape (nth i symbols) (nth (1+ i) symbols))
	     (setf (gethash (nth i symbols) allocator) (nth (1+ i) symbols)))
    
    (cl-waffe2/vm::adjust-allocation! (compiled-allocation model) allocator)
    nil))

(defmethod cl-waffe2/vm.nodes:forward ((model Compiled-Composite) &rest inputs &key &allow-other-keys)
  (let ((input-args (compiled-inputs model)))
    (when input-args

      (assert (= (length input-args) (length inputs))
	  nil
	  "forward: Can't invoke the forward step of given Compiled-Composite because the number of arguments received is invaild.
    (forward compiled-model~a)
                            └── the model is compiled as~a."
	  (with-output-to-string (out)
	    (dotimes (i (length inputs)) (format out " inputs[~a]" i)))
	  (with-output-to-string (out)
	    (dolist  (i input-args) (format out " ~a" i))))

      (loop for val in inputs
	    for name in input-args do
	      (set-input model name val))))
  
  ;; Check if all the inputs are embodied?
  (all-embodied? model)
  (let ((*runtime-mode-p* t))
    (with-adjustable-symbol-scope
      (set-adjustable-symbols model)
      (cl-waffe2/vm::with-static-allocation ((compiled-allocation model))
	(apply #'values
	       (map 'list #'cl-waffe2/vm.nodes::eliminate-undetermined-size
		    (multiple-value-list (funcall (compiled-forward model)))))))))

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
    (with-adjustable-symbol-scope
      (set-adjustable-symbols model)
      (cl-waffe2/vm::with-static-allocation ((compiled-allocation model))
	(funcall (compiled-backward model)))))
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
		(compile-mode :fastest)
		(fuse-ops t))
  "
## [function] build

```lisp
(build toplevel &key (inputs nil) (construct-backward? (not *no-grad*)) (compile-mode :fastest) (fuse-ops t))
```

Compiles the given computation node starting from `toplevel`. The docstring of `Compiled-Composite` describes how this function are used in practical.

### Inputs

`toplevel [AbstractTensor]` The end of node. Any shapes could be OK even when constructing backward.

`inputs[list]` Set a list of argument keywords here so that the method `forward` can receive arguments that have been lazily evaluated. The order is taken into account. (e.g.: Set to `(:A :B)` and forward can receive this: `(forward compiled-model (randn `(3 3)) (randn `(3 3)))`)

`construct-backward?` [boolean] Set t to build backward.

`compile-mode`[compile-mode-t] an keyword indicating the compiling option. (No significant impact on execution speed but compile speed. for any case `:fastest` is the best solution.)

`fuse-ops[boolean]` Set to enable `FusionOps` declared by `defpath`.
"
  (declare (type AbstractTensor toplevel))

  (when inputs
    (assert (every #'keywordp inputs)
	nil
	"build: Can't compile the tensor because the :inputs is malformed or invaild.
    (build toplevel :inputs ( ... ) ...)
                              └── :inputs receive a list of keyword indicating the name of tensors created by make-input
                                  Set like `(:A :B) "))
  
  (multiple-value-bind (fw-iseq bw-iseq variables dout allocation)
      (cl-waffe2/vm:compile-forward-and-backward toplevel
						 :need-backward construct-backward?
						 :compile-mode compile-mode
						 :fuse-p fuse-ops)
    (declare (ignore dout))

    (let ((forward-f  #'(lambda () (cl-waffe2/vm:accept-instructions fw-iseq)))
	  (backward-f   (when construct-backward? #'(lambda () (cl-waffe2/vm:accept-instructions bw-iseq))))
	  (table        (construct-variables-table variables)))


      ;; Check all arguments are valid as an argument
      (when inputs
	(let ((input-names (alexandria:hash-table-keys (nodevariables-variables table))))
	  (mapc
	   #'(lambda (x)
	       (when (not (find x input-names))
		 (error "build: Can't compile the tensor because the argument ~a didn't appear in the computation node.
        (build toplevel :inputs ~a)
                                └── Choose from: ~a "
			x
			inputs
			input-names)))
	   inputs)))

      (prog1
	  (make-instance 'Compiled-Composite
			 :allocation allocation
			 :compiled-forward  forward-f
			 :compiled-backward backward-f
			 :out               toplevel
			 :inputs            inputs
			 :variables         table)
	(mapc #'cl-waffe2/vm.nodes:on-finished-compiling *using-backend*)))))


;; TODO -> (defmethod free-model ((model Compiled-Composite)))

(defmethod print-object ((model Compiled-Composite) stream)
  (format stream "<Compiled-Composite
    forward  : ~a
    backward : ~a
~a>"
	  ;; Variables
	  (if (compiled-inputs model)
	      (with-output-to-string (out)
		(format out "forward(model")
		(dolist (i (compiled-inputs model)) (format out " ~a" i))
		(format out ") -> ~a{~a}~a" (class-name (class-of (compiled-out model))) (dtype (compiled-out model)) (shape (compiled-out model))))
	      (format nil "forward(model) -> ~a{~a}~a" (class-name (class-of (compiled-out model))) (dtype (compiled-out model)) (shape (compiled-out model))))
	  (if (compiled-backward model)
	      "backward(model) -> t"
	      "nil")
	  (if (>= (length (alexandria:hash-table-keys (nodevariables-variables (compiled-variables model)))) 1)
	      (compiled-variables model)
	      "")))


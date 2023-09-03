
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
  (let* ((syms  (nodevariables-symbols node))
 	 (vars  (nodevariables-variables node)) ;; Hash-Table
	 (input-keys (alexandria:hash-table-keys vars))
	 (table (make-print-table)))
    (format
     stream
     "+= [Tensors in the computation node] =======+

Subscripts:
~a

Variables:
~a

 - The number of tmp variables : ~a
 - The number of parameters    : ~a
+========================================+" ;; TODO: Print Number of Parameters
     (with-output-to-string (out)
       (loop for k downfrom (length syms) to 0 by 2
	     if (nth k syms)
	       do (format out "     [~a -> ~a, max=~a]~%"
			  (nth k syms)
			  (or (nth (1+ k) syms) "?")
			  (or (gethash (nth k syms) (nodevariables-adjustable-symbol node)) "?"))))

     (with-output-to-string (out)
       (let ((first-row)
	     (second-row))
	 (push "NAMES" first-row)
	 (push "SIZE"  second-row)
	 (loop for k upfrom 0 below (length input-keys)
	       do (push (nth k input-keys) first-row)
		  (push (shape (gethash (nth k input-keys) vars)) second-row))
	 
	 (mapc #'(lambda (n s)
		   (addrow! table
			    (make-row `(,(format nil "~a" n)
					,(format nil "~a" s)))))
	       (reverse first-row) (reverse second-row))
	 (render-table table out)))
     (length (nodevariables-tmp-variables node))
     (length (nodevariables-parameters node)))))

(defun embody-input (nodevars variable-name actual-tensor)
  "
## [function] embody-input

Giving values to delay-evaluated tensors.
"
  (declare (type NodeVariables nodevars))
  
  (let ((input-tensor (gethash variable-name (nodevariables-variables nodevars))))
    
    (when (null input-tensor)
      (error "The InputTensor named ~a weren't appeared in the computation node" variable-name))

    (when (not (= (dims input-tensor) (dims actual-tensor)))
      (error "embody-input: ranks does not match: ~a and ~a" input-tensor actual-tensor))
    
    (let ((symbols-changed (make-hash-table)))
      (loop for place in (tensor-input-shape input-tensor)
	    for value in (shape actual-tensor)
	    for rank upfrom 0
	    if (and (not (symbolp place))
		    (not (= place value)))
	      do (error "embody-input: The ~ath rank is a fixed dimension.
So the corresponding shape must be ~a but got ~a.

Shapes: Input_Place <- Actual_Tensor
-----------------------------------------------
Shapes:   ~a~a  <-  ~a

input-tensor:
~a

Strides       : ~a
permute-order : ~a

actual-tensor:
~a

Strides       : ~a
permute-order : ~a
"
			rank value place
			variable-name
			(tensor-input-shape input-tensor)
			(shape actual-tensor)
			input-tensor
			(tensor-stride input-tensor)
			(tensor-permute-order input-tensor)
			actual-tensor
			(tensor-stride actual-tensor)
			(tensor-permute-order actual-tensor))
	    if (symbolp place)
	      do (setf (gethash place symbols-changed) value))
            
      ;; InputTensor <- Actual-Tensor
      (embody-actual-tensor input-tensor actual-tensor)

      ;; Apply hash-table
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

;; ==============================================================================
;; Kernel Constructor | General-Purpose APIs
;; ==============================================================================

(defun compile-forward-kernel (forward-iseq
			       variables
			       &key
				 (compile-mode :default))
  "
## [function] compile-forward-kernel
"

  (declare (type compile-option-t compile-mode)
	   (ignore compile-mode))
  
  (let* ((inputs (loop for var in variables
		       if (user-input-p var)
			 collect var))
	 (set-input-forms))
    ;; set-input-form .. collects adjustable shapes
    (mapc #'(lambda (input)
	      (loop for shape in (shape input)
		    for kth-dim upfrom 0
		    if (symbolp shape)
		      do (push (list shape kth-dim input) set-input-forms)))
	  inputs)

    (values
     ;; [FixME] Eliminate this compile in the future release.
     #'(lambda ()
	 (with-adjustable-symbol-scope
	   (loop for form in set-input-forms do
	     (register-adjustable-shape (car form) (nth (second form) (shape (third form)))))
	   (cl-waffe2/vm:accept-instructions forward-iseq)))
     variables
     set-input-forms)))

(defun compile-backward-kernel (toplevel backward-iseq &key (compile-mode :default) (set-input-forms))
  (declare (type compile-option-t compile-mode)
	   (ignore compile-mode))

  
  (when (some #'symbolp (shape toplevel))
    (error "Can't construct backward, because the shape of tensor is undetermined: ~a

Try again with: (with-no-grad ...) " (shape toplevel)))

  (if set-input-forms
      #'(lambda ()
	  (with-adjustable-symbol-scope
	    (loop for form in set-input-forms do
	      (register-adjustable-shape (car form) (nth (second form) (shape (third form)))))
	    (cl-waffe2/vm:accept-instructions backward-iseq)))
      #'(lambda () (cl-waffe2/vm:accept-instructions backward-iseq))))

;; ==========================================
;; General-Purpose APIs
;; ==========================================

(defclass Compiled-Composite ()
  ((compiled-forward :initarg :compiled-forward :type function :reader compiled-forward)
   (compiled-backward :initarg :compiled-backward :type (or null function) :reader compiled-backward)
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

"))

(defun all-embodied? (model)
  "Invokes an simple-error when model has still unembodied tensor"
  (declare (type Compiled-Composite model))
  (let ((vars (nodevariables-variables (compiled-variables model)))
	(unembodied))
    (loop for key being the hash-keys in vars using (hash-value val)
	  if (null (vec val))
	    do (push (cons key val) unembodied))

    (when unembodied
      (error "Can't call forward of the given model,
because there's still unembodied tensors:
~a"
	     (with-output-to-string (out)
	       (dolist (k unembodied) (format out "~% :~a - ~a Tensor." (car k) (shape (cdr k)))))))))


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
    (funcall (compiled-forward model))))

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
    (funcall (compiled-backward model))))

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
		(use-setinput-form nil)
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
  
  (multiple-value-bind (fw-iseq bw-iseq variables) (cl-waffe2/vm:compile-forward-and-backward toplevel :need-backward construct-backward? :compile-mode compile-mode :fuse-p fuse-ops)
    (multiple-value-bind (fw-function variables set-input-forms) (compile-forward-kernel fw-iseq variables :compile-mode compile-mode)
      ;; Vars - All Variables (including ChainTMP) used in forward.
      (let ((table (construct-variables-table variables)))

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
	    (values (make-instance 'Compiled-Composite
				   :out toplevel
				   :inputs inputs
				   :variables  table
				   :compiled-forward fw-function
				   :compiled-backward (when construct-backward?
							(compile-backward-kernel toplevel bw-iseq :compile-mode compile-mode :set-input-forms set-input-forms)))
		    
		    (when use-setinput-form set-input-forms))
	  (mapc #'cl-waffe2/vm.nodes:on-finished-compiling *using-backend*))))))


(defmethod print-object ((model Compiled-Composite) stream)
  (format stream "<Compiled-Composite
    forward  : ~a
    backward : ~a
~a
>"
	  ;; Variables
	  (if (compiled-inputs model)
	      (with-output-to-string (out)
		(format out "forward(model")
		(dolist (i (compiled-inputs model)) (format out " ~a" i))
		(format out ") -> ~a{~a}~a" (class-name (class-of (compiled-out model))) (dtype (compiled-out model)) (shape (compiled-out model))))
	      (format nil "forward(model) -> ~a{~a}~a" (class-name (class-of (compiled-out model))) (dtype (compiled-out model)) (shape (compiled-out model))))
	  (if (compiled-backward model)
	      "backward(model)"
	      "nil")
	  (if (>= (length (alexandria:hash-table-keys (nodevariables-variables (compiled-variables model)))) 1)
	      (compiled-variables model)
	      "")))

;;
;; [fw, bw, leaves = build(toplevel)]
;;

;;
;; [TODO]
;;  diff! compute!
;; (let ((a (compute! top)))
;;         (diff! a))) ...
;;
;;


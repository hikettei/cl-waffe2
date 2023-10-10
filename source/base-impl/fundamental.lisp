
(in-package :cl-waffe2/base-impl)


;; ===============================================================
;; Copying APIs
;; ===============================================================

(defnode (MoveTensorNode (myself dtype &key (save-for-backward nil) (maybe-in-place nil))
	  :where (A[~] B[~] -> A[~])
	  :slots ((ignore-me :initform nil :accessor movetensor-ignore-me :type boolean)
		  (maybe-in-place :initform nil :initarg :maybe-in-place :type boolean :reader move-maybe-in-place)
		  (internal-lazy-save-for-backward :initform nil :accessor mv-lazy-sv4bw :type boolean)
		  (save-for-backward :initarg :save-for-backward :accessor movetensor-save-for-backward :type boolean)) ;; when t, ignored.
	  
	  :backward ((self dout dx dy)
		     (let ((dy-out
			     (if (and
				  (eql (tensor-attribute dy) :chain)
				  (movetensor-ignore-me self))
				 dout
				 (if (tensor-permuted-p dout)				     
				     (let ((out (make-input (shape dx) nil
							    :create-from dout
							    :dtype (dtype dx)
							    :order (order dx))))
				       (!move out dout :force t))
				     (!copy dout :force t)))))
		       (values nil dy-out)))
	  :documentation "
Moves all the visible elements of `B` into visible areas of `A`.

```math
A\\gets{B}
```

### Constructor

`(MoveTensorNode dtype)`

`dtype` dtype to use.

"))

(defnode (MoveScalarTensorNode (myself &key (save-for-backward nil) (maybe-in-place nil))
	  :out-scalar-p t
	  :slots ((ignore-me :initform nil :accessor movetensor-ignore-me :type boolean)
		  (maybe-in-place :initform nil :initarg :maybe-in-place :type boolean :reader move-maybe-in-place)
		  (internal-lazy-save-for-backward :initform nil :accessor mv-lazy-sv4bw :type boolean)
		  (save-for-backward :initarg :save-for-backward :accessor movetensor-save-for-backward :type boolean)) ;; when t, ignored.
	  
	  :where (A[scal] B[scal] -> A[scal] where scal = 1)
	  :backward ((self dout dx dy)
		     (let ((dy-out
			     (if (and
				  (eql (tensor-attribute dy) :chain)
				  (movetensor-ignore-me self))
				 dout
				 (!copy dout :force t))))		       
		       ;; dx/dy never shares pointer, so just moving to dx/dy is enough i guess.
		       
		       (values
			(if (eql (tensor-attribute dx) :chain)
			    (!move dx dout :force t)
			    dout)
			(if (eql (tensor-attribute dy) :chain)
			    (!move dy dy-out :force t)
			    dy-out))))))

(define-impl (MoveScalarTensorNode :device ScalarTensor)
	     :forward ((self x y)
		       `(if (not (movetensor-ignore-me ,self))
			    (progn
			      (setf (tensor-vec ,x) (tensor-vec ,y))
			      ,x)
			    ,y)))

(declaim (ftype (function (AbstractTensor AbstractTensor &key (:force boolean) (:maybe-in-place boolean)) (values AbstractTensor &optional))
		!move))
(defun !move (place tensor &key (force nil) (maybe-in-place nil))
  "
## [function] !move

```lisp
(!move place tensor &key (force nil) (maybe-in-place nil))
```

```math
A\\gets{B}
```

The function `!move` moves all the visible elements of tensor into all the visible elements of place.

### nodes

one of: `MoveTensorNode` `ScalarTensorNode`

### Inputs

`place[AbstractTensor]` tensor to be overwritten.

`tensor[AbstractTensor]` tensor to be referred.

`force[boolean]` If, pruning/in-place-mutation by compilers aren't applied

`maybe-in-place[boolean]` Set T to ignore the copy; the operation is replaced with the function just returning `place`. Moves with this parameter, is displayed as `ALLOC{INTENRAL}` when disassembled.

### Output

`Tensor[AbstractTensor]`
"
  (if (and (scalar-p place)
	   (scalar-p place))
      (forward (MoveScalarTensorNode :save-for-backward force) place tensor)
      ;; The problem is that: it is unknown whether place or tensor is returned until optimize-computation-node! is called.
      (forward (MoveTensorNode (dtype place) :save-for-backward force :maybe-in-place maybe-in-place)
	       place
	       tensor)))

(declaim (ftype (function (AbstractTensor &key (:force boolean) (:maybe-in-place boolean)) (values AbstractTensor &optional))
		!copy))
(defun !copy (tensor &key (force nil) (maybe-in-place nil))
  "
## [function] !copy

```lisp
(!copy tensor &key (force nil) (maybe-in-place nil))
```

The function !copy makes a clone of given tensor which is InputTensor, and moves the elements of tensor into the new tensor. broadcasted elements are keep broadcasted (if you want to create contiguous tensors, use `->contiguous`). Copies are prone to bottlenecks in the network, so a lot of special optimisation is applied. If you want to exclude it, set `:force` to t. Thanks to such optimisations, unlike other libraries, this function is used to create a temporary region of Tensor.

```lisp
(defun my-add (a b)
    (call (AddNode :float) (!copy a) b))
```

In this case, the my-add function can be used as a function without side effects. However, after compilation, any unneeded copies are removed.

If the value of tensor is immediately overwritten and the element does not need to be copied, then `:maybe-in-place` should be set to T. And, the elements of retuend tensor is filled random because it is brought from memory-pool.

Input:  `Tensor[AbstractTensor]`
Output: `Tensor[AbstractTensor]`
"
  (let* ((out (make-input (actual-shape tensor) nil
			  :create-from tensor
			  :scalar-p (scalar-p tensor)
			  :dtype (dtype tensor)
			  :order (order tensor)))
	 (broadcasted-p)
	 (broadcasts (loop for size in (shape tensor)
			   for view in (tensor-view tensor)
			   if (eql :broadcast (viewtype (force-list view)))
			     collect (and
				      (setq broadcasted-p t)
				      `(:broadcast ,size))
			   else
			     collect t))
	 (out (if broadcasted-p
		  (apply #'!view out broadcasts)
		  out))
	 (res (!move out tensor :force force :maybe-in-place maybe-in-place)))
    ;; Extend flexible-p, because !copy is used to make a cache before using basic-function like !add
    (extend-states res tensor)))


;; ===============================================================
;; View APIs
;; ===============================================================

;; Both !view and !reshape has the same format of arguments:
;; (function tensor &rest args)

(defnode (ViewTensorNode (myself subscripts result before)
	  :slots ((subscripts :initarg :subscripts))
	  :where (A[result] B[before] -> A[result]))
  (setf (ignore-shape-error myself) t))

(define-impl (ViewTensorNode)
	     :forward
	     ((self viewed-tensor old)
	      `(progn
		 (setf (tensor-vec ,viewed-tensor) (tensor-vec ,old))
		 ,viewed-tensor))
	     :backward
	     ;; [FixME] Slicing backward won't works
	     ;; e.g.:
	     ;; randn(3 3)[0:2, t].backward()
	     ;; (1 1 1)
	     ;; (1 1 1)
	     ;; (0.1 0.2 0.3) <- should be filled with 0
	     ((self dout dx dy) ;; (viewed-tensor old)
	      (let* ((out-sub (tensor-view dy))
		     (inp-sub (slot-value self 'subscripts))
		     (res (!move dx (apply #'!view dout inp-sub))))		
		(values nil (->contiguous (apply #'!view res out-sub))))))

(defun !view (tensor &rest subscripts)
  "
## [function] !view

```lisp
(!view tensor &rest subscripts)
```

The function !view returns a tensor which is applied lazy-evaluated view.

For Example, let A be a 4x8 Matrix, and we gonna create a view of A that portrays `A[:, 2]`.

```
(!view A 2 t)

     A                            B
0 ++++++++                     --------
1 ++++++++                     --------
2 ++++++++ -> [make a view] -> ++++++++
3 ++++++++                     --------
```

Here,

1. `A` and `B` shares the pointer.

2. Calling `(shape B)` returns `(1 8)`.

### Subscripts

Subscripts are following:

1. `t` all elements in the axis.

2. `fixnum` points out the specified index.

3. `(start end)` slices the area.

4. `(start end step-by)` slices the area by `step-by`. step-by can be a negative-fixnum. (Not tested)

5. `(:broadcast N-times)` broadcasts the axis for N-times, the axis to be broadcasted must be 1 or broadcasted-axis.

6. `(:tflist ...)` (TODO)

7. `(:indices ...)` (TODO)

### Return

`(values sliced-tensor broadcast-reverser)`

Tips: Applying `!view` again to the returned `sliced-tensor` with `broadcast-reverser` will remove broadcasts from the tensor.

Tips: If a function is passed as the first element of `subscript`, the subscript is overwritten based on the return value of the function. The function is called like: `(funcall function tensor)` can be used like: `(!view tensor (compose #'reverse #'tensor-view))`.
"
  (let* ((subscripts (if (functionp (car subscripts))
			 (funcall (car subscripts) tensor)
			 subscripts))
	 (out (apply #'cl-waffe2/vm.generic-tensor::view tensor subscripts))
	 (broadcast-reverser
	   (loop for s in (tensor-view out)
		 if (and (listp (force-list s))
			 (eql (car (force-list s)) :broadcast))
		   collect 0
		 else
		   collect t)))
    ;; Update Chains
    (values
     (forward (ViewTensorNode subscripts (shape out) (shape tensor)) out tensor)
     broadcast-reverser)))


(defnode (ReshapeTensorNode (self before after)
	  :where (A[before] B[after] -> B[after])
	  :slots ((before :initarg :before :reader reshapenode-shape))
	  :backward ((self dout dx dy)
		     (declare (ignore dx dy))
		     (values (apply #'!reshape dout (reshapenode-shape self)) nil))
	  :documentation "")
  (setf (ignore-shape-error self) t))

(define-impl-op (ReshapeTensorNode :device t)
		:forward ((self x y)
			  ;; Reshaping is the operation where:
			  ;;  The order of storage vec is the same.
			  ;;  But other factors (e.g.: Shaping, Strides)
			  ;;  Has Changed.

			  ;; [TODO] Detect This Error Before Execution.
			  (assert (= (total x) (total y))
				  nil
				  "ReshapeTensorNode: Attempted to move x to y but failed because the total sizes considering dynamically shape do not match:
~a and ~a" x y)
			  (setf (tensor-vec y) (tensor-vec x))
			  y))

;; ===============================================================
;; Reshaping APIs
;; ===============================================================

(defun parse-reshape-args (before-shape after-shape)
  "check after-shape is consisted of positive fixnum.
shapes can contain t at once, this function also infers t."

  (when (some #'(lambda (x) (and (not (eql x t))
				 (or
				  (cl-waffe2/vm::lazyaxis-p x)
				  (symbolp x))))
	      after-shape)
    (when (some #'(lambda (x) (eql x t)) after-shape)
      (error "!reshape: can't infer the value of t because adjustable shapes and `t` cant used in the same time."))
    (return-from parse-reshape-args after-shape))
  
  (assert (<= (count t after-shape) 1)
	  nil
	  "!reshape: Assertion Failed because t only appears at once.")

  (assert (every #'(lambda (x)
		     (or (eql x t)
			 (> x 0)))
		 after-shape)
	  nil
	  "!reshape: Assertion Failed because shapes aren't consisted of positive fixnum.")

  (let* ((without-t (loop for s in after-shape unless (eql s t) collect s))
	 (t-inferred (/ (apply #'* before-shape) (apply #'* without-t))))
    (loop for s in after-shape
	  if (eql s t)
	    collect t-inferred
	  else
	    collect s)))

(declaim (ftype (function (AbstractTensor &rest (and (not null) (or function boolean fixnum))) AbstractTensor) !reshape))
(defun !reshape (tensor &rest shapes)
  "
## [function] !reshape

```
(!reshape tensor &rest shapes)
```

Changes the shape of given tensor.

Before and after the operation, the total elements of tensors must correspond.

### Inputs

`tensor` `AbstractTensor` but must not includes `symbol` in the shape.


`shapes` could be one of: fixnum `t`. `t` can be used at one, but the value of t is automatically inferenced.

Note: If the first element of `shapes` is a function, `shapes` are overwritten with the function's value.

```lisp
(!reshape (ax+b `(5 3 2) 1 0) (compose #'reverse #'shape)) ;; => (2 3 5) Tensor
```
"
  (declare (type AbstractTensor tensor))
  
  (let* ((shapes (if (functionp (car shapes))
		     (funcall   (car shapes) tensor)
		     (loop for s in shapes
			   if (or (eql s t)
				  (numberp s))
			     collect s
			   else
			     collect (cl-waffe2/vm:make-lazyaxis s))))
	 (shapes (parse-reshape-args (shape tensor) shapes))
	 (result (make-input shapes nil
			     :dtype (dtype tensor)
			     :order (order tensor))))

    (when (and (every #'numberp (shape result))
	       (every #'numberp shapes))
      (assert (= (apply #'* (shape tensor))
		 (apply #'* shapes))
	      nil
	      "Reshaping failed because the total sizes do not match."))
    ;; (!view tensor `(2 4) `(2 4)) -> Copy
    ;; (!view tensor  0 t t t)
    (let ((out (forward (ReshapeTensorNode (shape tensor) shapes)
			(->contiguous tensor)
			result)))
      out)))

;; !squeeze/!unsqueeze

;; TO ADD: (defun !lazy-reshape (tensor &rest shapes) ) reshape but can include symbol as shapes

;; Memo:
;; The behaviour of ScalarTensor is ugly? because...
;; (!sum tensor).shape   = (1)
;; (make-tensor 1).shape = (1)

(with-export !flatten
  (defun !flatten (tensor)
    "
## [function] !flatten

```
(!flatten tensor)
```

equivalent to the `(!reshape tensor t)`
"
    (!reshape tensor t)))

(declaim (ftype (function (AbstractTensor fixnum &key (:at fixnum)) AbstractTensor) !rankup))
(defun !rankup (tensor ntimes &key (at 0))
  "
## [function] !rankup

```lisp
(!rankup tensor ntimes &key (at 0))
```

The function !rankup appends/reduces 1 at `at` into the given tensor's shape for ntimes.

1. If `ntimes` > 0, appends 1

2. If `ntimes` < 0, reduces 1, if the axis=1, otherwise returns error.

### Examples

```lisp
CL-WAFFE2-REPL> (!rankup (randn `(3 3)) 3 :at 1)
{CPUTENSOR[float] :shape (3 1 1 1 3) :named ChainTMP1459457 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (3 1 1 1 3) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: RESHAPETENSORNODE-T (A[BEFORE] B[AFTER] -> B[AFTER])>}
CL-WAFFE2-REPL> (!rankup * -3 :at 1)

{CPUTENSOR[float] :shape (3 3) :named ChainTMP1459467 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (3 3) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: RESHAPETENSORNODE-T (A[BEFORE] B[AFTER] -> B[AFTER])>}
CL-WAFFE2-REPL>
```
"
  (declare (type AbstractTensor tensor)
	   (type fixnum ntimes at))
  (let* ((at (if (>= at 0)
		 at
		 (+ (dims tensor) at)))
	 ;; (leading-shape 1 ... 1 last-shape)
	 (leading-shape (subseq (shape tensor) 0 at))
	 (shape         (nthcdr at (copy-list (shape tensor)))))
    (if (< ntimes 0)
	(loop for i fixnum upfrom 0 below (abs ntimes)
	      do (if (= (car shape) 1)
		     (pop shape)
		     (error "!rankup failed because it encountered a dimension which is not the equivalent to 1.")))
	(loop for i fixnum upfrom 0 below ntimes
	      do (push 1 shape)))
    ;; TODO: view broadcast
    (apply #'!reshape tensor `(,@leading-shape ,@shape))))

(defnode (Mat->ScalarNode (myself)
	  :out-scalar-p t
	  :where (Matrix[~ scal] Scalar[scal] -> Scalar[scal] where scal = 1)
	  :backward ((self dout dm ds)
		     (declare (ignore dm ds))
		     (values
		      (->mat dout)
		      nil))))

(defnode (Scalar->MatNode (myself out-shape)
	  :where (Scalar[scal] Matrix[~ scal] -> Matrix[scal] where scal = out-shape)
	  :backward ((self dout ds dm)
		     (declare (ignore dm ds))
		     (values (->scal dout) nil))))

(define-impl (Mat->ScalarNode :device t)
	     :forward ((self matrix scalar)
		       `(progn
			  (setf (tensor-vec ,scalar) (vref ,matrix 0))
			  ,scalar)))

(define-impl (Scalar->MatNode :device t)
	     :forward ((self scalar matrix)
		       `(progn
			  (tensor-vec ,matrix) ;; Call Lazy-Allocate of matrix
			  (setf (vref ,matrix 0) (tensor-vec ,scalar))
			  ,matrix)))

;; Add: Docstring
;; Add: Shape Check
(with-export ->scal
  (defun ->scal (matrix-tensor)
    "
## [function] ->scal

```
(->scal matrix-tensor)
```

The function ->scal receives `matrix-tensor` with total-size = 1, returning a ScalarTensor.
"
    (forward (Mat->ScalarNode)
	     (!reshape matrix-tensor 1)
	     (make-input `(1)
			 nil
			 :scalar-p t
			 :dtype (dtype matrix-tensor)))))

(with-export ->mat
  (defun ->mat (scalar-tensor &key (dims 1))
    "
## [function] ->mat

```
(->mat scalar-tensor &key (dims 1))
```

The function ->mat receives `ScalarTensor`, returning a matrix with the number of axis=dims."
    (let ((out-shape (make-list dims :initial-element 1)))
      (forward (Scalar->MatNode out-shape)
	       scalar-tensor
	       (make-input out-shape nil
			   :dtype (dtype scalar-tensor))))))

		       

;; ===============================================================
;; Proceed APIs
;; ===============================================================

;; The definition of value node is dynamically changed and redefined.
;; Forward  -> All The Previous Forward Steps
;; Backward -> All The Previous Backward Steps.

;; We can also add: Proceed-Auto

(defnode (ProceedNode (myself toplevel &key (measure-time nil) (compile-mode :default))
	  :where (A[~] -> A[~])
	  :slots ((measure-time   :initarg :measure-time :reader measure-time-p)
		  (compiled-model :initform nil :accessor proceed-compiled-model)
		  (compile-mode   :initarg :compile-mode :reader compile-mode)
		  (result         :accessor proceed-result))
	  :documentation "ProceedNode is a special node which takes all the previous computation node before tensor.")

  (let ((compiled-model (if measure-time
			    (progn
			      (format t "[proceed-time] build ->~%")
			      (time (build toplevel :compile-mode compile-mode)))
			    (build toplevel :compile-mode compile-mode :construct-backward? (and (not *no-grad*) (ancestor-param-p toplevel))))))
    ;; Detaching the tensor
    (setf (detach-p toplevel) T
	  (proceed-compiled-model myself) compiled-model)))
		

(define-impl (ProceedNode :device t)
	     :save-for-backward (nil)
	     :forward ((self x)
		       (let ((compiled-model (proceed-compiled-model self)))			 
			 (if (measure-time-p self)
			     (progn
			       (format t "[proceed-time] With allocation time:~%")
			       (time (forward compiled-model))
			       (format t "[proceed-time] Without allocation time:~%")
			       (setf (proceed-result self) (time (forward compiled-model))))
			     (setf (proceed-result self) (forward compiled-model)))
			 
			 ;; Tell cl-waffe2 VM the returned value's type
			 (setf (out-scalar-p self) (scalar-p (proceed-result self)))
			 
			 ;; The result is returned.
			 `(progn
			    ;; Tell top compiling funtion what composite to use for the compiled-function			    
			    ,x)))
	     :backward ((self dout dx)
			(declare (ignore dx))
			(let ((compiled-model (proceed-compiled-model self)))
			  (values
			   (with-instant-kernel dout
			     `(and
			       ,(if (measure-time-p self)
				    `(progn
				       (format t "[proceed-time] Reverse Mode~%")
				       (time (backward ,compiled-model)))
				    `(backward ,compiled-model))
			       ;; Delete Gradients.
			       (!mul 0 ,dout)))))))

;; TODO: ProceedNode for several outputs
(defun proceed (tensor &key (measure-time nil) (compile-mode :default))
  "
## [function] proceed

```
(proceed tensor &key (measure-time nil))
```

The function proceed invokes special node, `ProceedNode`, which takes all the previous computation node before tensor, returning the result of it.

The backward is created with the previous node.


This function will be useful especially when debugging on REPL.

### Inputs

If `measure-time`=t, ProceedNode wraps with time macro when calling **COMPILED** forward and backward propagation. Compiling time isn't included to the displayed time while (time (proceed tensor)) includes.

`compile-mode` is a keyword, type of `compile-mode-t`.
"
  (let* ((node (ProceedNode tensor :measure-time measure-time :compile-mode compile-mode))
	 ;; Previous Node is already compiled, so detach tensor from nodes.
	 (out  (forward node tensor)))
    
    ;; Out is still unallocated, so set the result.
    (if (scalar-p out)
	(setf (tensor-vec out) (tensor-vec (proceed-result node)))
        (embody-actual-tensor out (proceed-result node)))
    out))

(defun proceed-time (tensor &key (compile-mode :default))
  "
## [function] proceed-time

```
(proceed-time tensor)
```

An alias for (proceed tensor :measure-time t)

Note that: the proceed-time function invokes forward function twice times, in order for processing system to trace compiled lisp code, and ignoring allocation time."
  (declare (type AbstractTensor tensor))
  (proceed tensor :measure-time t :compile-mode compile-mode))

(defun proceed-backward (tensor &key (compile-mode :default))
  "
## [function] proceed-backward

```
(proceed-backward tensor)
```

The function proceed-backward calls forward and backwrd of the tensor.

### Output

`T` (which indicates backward is succeed)
"
  (declare (type AbstractTensor tensor))
  (let ((compiled-model (build tensor :compile-mode compile-mode)))
    (forward compiled-model)
    (backward compiled-model)))

(defun proceed-bench (tensor &key			      
			       (compile-mode :default)
			       (n-sample 1)
			       (ignore-first-call nil)
			       (stream t)			       
			       (top-k 10)
			       (backward nil)
			       (fuse-p t))
  "
## [function] proceed-bench

```lisp
(proceed-bench tensor &key (compile-mode :default) (n-sample 1) (ignore-first-call nil) (stream t) (top-k 10) (backward nil) (fuse-p t))
```

Invokes `cl-waffe2 VM` with benchmarking the forward and (if specified) backward.

### Input

`backward[boolean]` Set t in order to profile backward.

### Example

```lisp
CL-WAFFE2-REPL> (proceed-bench (!sum (randn `(3 3))))
 Time(s) |   Instruction ( * - Beyonds the average execution time)
2.3e-4*  | <WfInst[Compiled: SCALARMUL-CPUTENSOR] : TID1389503 <= op(TID1389503(1 1) <Input>TID1389505(1))>
2.0e-6   | <WfInst[Compiled: VIEWTENSORNODE-T]    : TID1389514 <= op(TID1389514(3 3) TID1389503(1 1))>
7.0e-6   | <WfInst[Compiled: ADDNODE-CPUTENSOR]   : TID1389514 <= op(TID1389514(3 3) <Input>TID1389488(3 3))>
1.0e-6   | <WfInst[Compiled: VIEWTENSORNODE-T]    : TID1389536 <= op(TID1389536(1 1) TID1389514(3 3))>

4 Instructions | 5 Tensors

 Total Time: 2.4e-4 sec

 Instruction                           | Total time (s) | Time/Total (n-sample=1)
<WfInst[Compiled: SCALARMUL-CPUTENSOR] | 2.3e-4 | 95.833336%
<WfInst[Compiled: ADDNODE-CPUTENSOR]   | 7.0e-6 | 2.916667%
<WfInst[Compiled: VIEWTENSORNODE-T]    | 3.0e-6 | 1.2500001%
{CPUTENSOR[float] :shape (1 1) -> :view (<(BROADCAST 1)> <(BROADCAST 1)>) -> :visible-shape (1 1) :named ChainTMP1389502 
  ((-0.43719095))
  :facet :input
  :requires-grad NIL
  :backward NIL}
```
"

  (multiple-value-bind (fw-iseq bw-iseq leaves dout allocation)
      (cl-waffe2/vm:compile-forward-and-backward tensor :compile-mode compile-mode :fuse-p fuse-p)
    (declare (ignore leaves dout))
    (let ((cl-waffe2/vm.generic-tensor::*runtime-mode-p* t))
      (cl-waffe2/vm.generic-tensor::with-adjustable-symbol-scope
	(cl-waffe2/vm::with-static-allocation (allocation)
	  (let ((result))
	    (setq result
		  (cl-waffe2/vm:benchmark-accept-instructions fw-iseq
							      :n-sample n-sample
							      :ignore-first-call ignore-first-call
							      :stream stream
							      :top-k top-k))

	    (when backward
	      (format stream "[Benchmarking backward] ... ~%")
	      (cl-waffe2/vm:benchmark-accept-instructions bw-iseq
							  :n-sample n-sample
							  :ignore-first-call ignore-first-call
							  :stream stream
							  :top-k top-k))
	    result))))))

;; ===============================================================
;; Broadcast APIs
;; ===============================================================

(defnode (Flexible-Rank-Node (myself At)
	  :where (A[~] -> A[~])
	  :slots ((at :initarg :at :reader flex-at))
	  :backward ((self dout dx)
		     (declare (ignore dx))
		     (values (!flexible dout :at (flex-at self))))))

(define-impl (Flexible-Rank-Node :device t) :forward ((self x) `(progn ,x)))

(defun !flexible (tensor &key (at 0))
  "
## [function] !flexible

```
(!flexible tensor)
```

The function !flexible inserts a `broadcastable axes` to the tensor at the given position `at` (specified like: 1 2 ... -1 -2 ...).

That is:

```
Tensor = (10 10) -> [!flexible] -> Tensor' = (1 ... 1 10 10)
                                                 ^ <1 x N>
```

Note that added axes could be broadcasted automatically when the operation called with multiple arguments.

### Example

`!flexible` is a fundamental operation when using broadcasting in cl-waffe2. And usually called via `%transform` macro for readability.

```lisp
CL-WAFFE2-REPL> (!add (ax+b `(3 3) 0 0) (print (!flexible (ax+b `(3) 1 0) :at -1)))

{CPUTENSOR[float] :shape (3 <1 x N>) :named ChainTMP1631118 
  :vec-state [maybe-not-computed]
  (0.0 1.0 2.0)
  :facet :input
  :requires-grad NIL
  :backward <Node: FLEXIBLE-RANK-NODE-T (A[~] -> A[~])>} 
{CPUTENSOR[float] :shape (3 3) :named ChainTMP1631165 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (3 3) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>}
CL-WAFFE2-REPL> (proceed *)
{CPUTENSOR[float] :shape (3 3) :named ChainTMP1631189 
  :vec-state [computed]
  ((0.0 0.0 0.0)
   (1.0 1.0 1.0)
   (2.0 2.0 2.0))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
CL-WAFFE2-REPL> (!add (ax+b `(3 3) 0 0) (print (!flexible (ax+b `(3) 1 0))))

{CPUTENSOR[float] :shape (<1 x N> 3) :named ChainTMP1631205 
  :vec-state [maybe-not-computed]
  (0.0 1.0 2.0)
  :facet :input
  :requires-grad NIL
  :backward <Node: FLEXIBLE-RANK-NODE-T (A[~] -> A[~])>} 
{CPUTENSOR[float] :shape (3 3) :named ChainTMP1631248 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (3 3) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>}
CL-WAFFE2-REPL> (proceed *)
{CPUTENSOR[float] :shape (3 3) :named ChainTMP1631272 
  :vec-state [computed]
  ((0.0 1.0 2.0)
   (0.0 1.0 2.0)
   (0.0 1.0 2.0))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```
"
  (declare (type fixnum at))
  (let ((out (forward (Flexible-Rank-Node At) tensor))
	(at (if (>= at 0)
		at
		(+ 1 (dims tensor) at))))
    (when (or (not (>= at 0))
	      (> at (dims tensor)))
      (error "!flexible: can't add an broadcastable axis to the position ~a,"
	     at))
    (setf (tensor-flexible-p out) at)
    out))


;; ===============================================================
;; Logging
;; ===============================================================


(define-and-impl-node (PrintNode (self stream print-result print-dout mark)
		       :slots ((stream :initform nil :initarg :stream :reader print-stream)
			       (print-result :initform nil :initarg :print-result :reader print-result)
			       (print-dout :initform nil :initarg :print-dout :reader print-dout)
			       (mark       :initform nil :initarg :mark :reader print-mark))
		       :where (A[~] -> A[~])
		       :forward ((self x)
				 (if (print-result self)
				     `(locally (declare (optimize (speed 1)))
					(format
					 (print-stream ,self)
					 "~%===> [Forward] PrintNode: ~a ~a =========>~%~a" ,self ,(print-mark self) ,x)
					,x)
				     `(progn ,x)))
		       :backward ((self dout dx)
				  (declare (ignore dx))
				  (if (print-dout self)
				      (values
				       (with-instant-kernel dout
					 `(locally (declare (optimize (speed 1)))
					    (format
					     (print-stream ,self)
					     "~%<== [Backward] PrintNode: ~a ~a <========
Previous dout:
~a"
					     ,self
					     ,(print-mark self)
					     ,dout)
					    ,dout)))
				      (values dout)))))

(defun lazy-print (tensor
		   &key
		     (stream t)
		     (result t)
		     (dout t)
		     (mark ""))
  "
## [function] lazy-print

result ... result
dout   ... dout values"

  (forward (PrintNode stream result dout mark) tensor))

;; ===============================================================
;; Permute APIs
;; ===============================================================

(defun permute-backward-order (old-order new-order)
  (loop with rank = (1- (length old-order))
	for tgt in old-order
	collect (- rank (position tgt new-order))))

(define-and-impl-node (Permute-Node (self before after permute-old)
		       :slots ((permute-old :initform nil :initarg :permute-old :reader permute-old))
		       :where (Old[before] New[after] -> New[after])
		       :forward ((self a out)
				 `(let ((out1 (cl-waffe2/vm.generic-tensor::detach-and-clone1 ,out)))
				    ;;(embody-actual-tensor
				    ;; out1
				    ;; ,a)
				    (setf (tensor-vec out1) (tensor-vec ,a))
				    out1))
		       :backward ((self dout a out)
				  ;;(print (cl-waffe2/vm.generic-tensor::tensor-permute-order a))
				  ;;(print (cl-waffe2/vm.generic-tensor::tensor-permute-order dout))
				  (let* ((result
					   (apply #'!permute
						  dout
						  ;; dout.order and a.order -> out.order
						  (permute-backward-order
						   (loop for i fixnum downfrom (dims a) to 1 collect (1- i))
						   (cl-waffe2/vm.generic-tensor::tensor-permute-order out)))))
				    (values result nil))))
  (setf (ignore-shape-error self) t))

(defun list-diff (lista listb)
  (loop for l1 in lista
	for l2 in listb
	collect (= l1 l2)))

;; [Fix] Update description
(defun !permute (tensor &rest orders)
  "
## [function] !permute

In cl-waffe2, each tensor has a slot `(tensor-permute-order tensor)`, which indicates the order of the dimensions to be invoked. The function `!permute` returns a view of the original tensor input with its dimensions permuted.

```lisp
(n) (n-1) ... (1) (0) ... The order

 ++++   ^ (0)
 ++++   |
 ++++   |
        |
 ----> (1)

(A beautiful figure would be displayed in the future :<)
```

In other view, `!permute` replaces the order of following operation:

```lisp
A = 2x2x2 Matrix.

------------------------
Shape      :   2  2  2
Stride     :   4  2  1
[Permution]:   2  1  0
             A[1][1][1]
------------------------
```

When `[Permution]` is shuffled, the order of other parameters (e.g.: `shape` `stride` `view`...) are shuffle in tandem. That is, if we give `2 0 1` as a permutation, the figure becomes:

```lisp
A = 2x2x2 Matrix.

------------------------
Shape      :   2  2  2
Stride     :   4  1  2
[Permution]:   2  0  1
             A[1][1][1]
------------------------
```

The operation could be applied to transpose matrices.

### Example

```lisp
(defun transpose-revisit (tensor)
    ;; A[i j] -> A[j i]
    (!permute tensor :~ 0 1))
```

Note that the case when only the last two aces are subject to be swapped, we return `Lazy-Transpsose-Node` instead (for matmul).
### Inputs

`tensor[AbstractTensor]` tensor to be permuted.

`order[list<Fixnum>]` An list of permutation. Note that `:~` could be used once in an order If needed. If the order and the number of dimensions of the entered tensor do not match, the part is automatically stored as long as `:~` is provided.

Tips: If the first element of `order` arguments is a function, the rest arguments of `order` is overwritten with its result. that is, `order` become the value of `(funcall (car order) (tensor-permute-order tensor))` and can be used like: `(!permute tensor (compose #'reverse #'tensor-permute-order))` to reverse all permution for example.

Tips: `(!permute tensor (torch-order 2 1 0))` to use the same notation to pytorch.
"
  ;; If only the last two axes are subject to swapped.
  ;; Return a special node LazyTranspose instead.

  ;; BugCode:
  ;; (let ((a (parameter (print (ax+b `(3 10) 1 0)))))
  ;;		  (proceed-backward (!matmul a (randn `(10 3))))
  ;;		  a)
  ;;
  (let* ((orders (if (functionp (car orders))
		     (funcall (car orders) tensor)
		     orders))
	 (new-tensor (apply #'permute* tensor orders))
	 (diff       (list-diff (cl-waffe2/vm.generic-tensor::tensor-permute-order tensor)
				(cl-waffe2/vm.generic-tensor::tensor-permute-order new-tensor)))
	 (lazy-p (and (every #'(lambda (x) x) (butlast diff 2))
		      (every #'null           (last    diff 2))))
	 (out  (forward (Permute-Node
			 (shape tensor)
			 (shape new-tensor)
			 (cl-waffe2/vm.generic-tensor::tensor-permute-order new-tensor))
			tensor
			new-tensor)))
    ;; The case when (T NIL NIL) (T T NIL NIL) (NIL NIL) ... subject to lazy-transpose
    
    ;; judge: the diff is last two?
    (if lazy-p
	(call (LazyTransposeNode) out)
	out)))


(defun ->contiguous (tensor &aux (permuted-p (tensor-permuted-p tensor)))
  "
## [function] ->contiguous

Returns a copy of the given tensor if is is permuted. Otherwise returns the argumement as it is.

A memory-layout of returned copies are arranged into the same array as the array seen on the REPL.

### Example

```lisp
(!t (ax+b `(3 3) 1 0))

{CPUTENSOR[float] :shape (3 3) -> :view (<T> <T>) -> :visible-shape (3 3) :named ChainTMP110110 
  :vec-state [maybe-not-computed]
  ((0.0 3.0 6.0)
   (1.0 4.0 7.0)
   (2.0 5.0 8.0))
  :facet :input
  :requires-grad NIL
  :backward <Node: LAZYTRANSPOSENODE-T (A[~ I J] -> A[~ I J])>}

(tensor-vec *)

#(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)


;; calling ->contiguous...

(->contiguous (!t (ax+b `(3 3) 1 0)))
{CPUTENSOR[float] :shape (3 3) :named ChainTMP110149 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (3 3) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: MOVETENSORNODE-CPUTENSOR (A[~] B[~] -> A[~])>}

(tensor-vec (proceed *))
#(0.0 3.0 6.0 1.0 4.0 7.0 2.0 5.0 8.0)
```
"
  (if (or (tensor-projected-p tensor)
	  permuted-p)
      ;; ScalarTensor never becomes projected array
      (let* ((contiguous-place (make-input (shape tensor) nil
					   :dtype (dtype tensor)
					   :order (order tensor))))
	(!move contiguous-place tensor :force t))
      tensor))


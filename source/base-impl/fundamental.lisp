
(in-package :cl-waffe2/base-impl)


;; ===============================================================
;; Copying APIs
;; ===============================================================

(defnode (MoveTensorNode (myself dtype &key (save-for-backward nil))
	  :where (A[~] B[~] -> A[~])
	  :slots ((ignore-me :initform nil :accessor movetensor-ignore-me :type boolean)
		  (save-for-backward :initarg :save-for-backward :accessor movetensor-save-for-backward :type boolean)) ;; when t, ignored.
	  
	  :backward ((self dout dx dy)
		     (declare (ignore dx))
		     (let ((dy-out
			     (if (and
				  (eql (tensor-attribute dy) :chain)
				  (movetensor-ignore-me self))
				 dout
				 (!copy dout :force t))))
		       (values nil dy-out)))
	  :documentation "
Moves all the visible elements of `B` into visible areas of `A`.

```math
A\\gets{B}
```

### Behaviour

All cl-waffe2 operations follow this rule: `Make a copy for now, disable later`. (e.g.: the function `(!add x y)` makes an copy of `x` and `y` for now, but this copy operation is ignored, if they're concluded not to be needed, by tracing computation node.)

In order to disable a useless copy operations, MoveTensorNode must follow this behaviour:

1. Reading (movetensor-ignore-me self) in runtime, the forward makes a copy of given tensor only after the slot is `nil`.

2. Otherwise, return `B`

Don't worry the allocation won't be done until `(tensor-vec A)` is called.

For practical example, my impls (`./source/backends/lisp/arithmetic.lisp` for example) would be helpful!.

### Constructor

`(MoveTensorNode dtype)`

`dtype` dtype to use.
"))

(defnode (MoveScalarTensorNode (myself &key (save-for-backward nil))
	  :out-scalar-p t
	  :slots ((ignore-me :initform nil :accessor movetensor-ignore-me :type boolean)
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

(defun !move (place tensor &key (force nil))
  "
## [function] !move

```lisp
(!move place tensor)
```

```math
A\\gets{B}
```

The function !move returns a node which moves tensor's visible elements into place's visible elements.

### nodes

one of: `MoveTensorNode` `ScalarTensorNode`

### Inputs

`place[AbstractTensor]` tensor to be overwritten.

`tensor[AbstractTensor]` tensor to be referred.

`force[boolean]` If t, the pruning of operation by cl-waffe2 will never done.

### Output

Unevaluated Copied Tensor."
  (if (and (scalar-p place)
	   (scalar-p place))
      (forward (MoveScalarTensorNode :save-for-backward force) place tensor)
      ;; The problem is that: it is unknown whether place or tensor is returned until optimize-computation-node! is called.
      (forward (MoveTensorNode (dtype place) :save-for-backward force)
	       place
	       tensor)))


(defun !copy (tensor &key (force nil))
  "
## [function] !copy

```lisp
(!copy tensor)
```

The function !copy returns a node which makes a copy the tensor's visible area.

Note that: the function `!copy` never creates a new tensor larger than (tensor-vec tensor) has, (i.e.: copying broadcasted tensor will return broadcasted and copied tensor).

`!copy` is used to make a cache before calling destructive operation to avoid side effects, therefore if the copy is included to be useless by compiler, this operations is being ignored without changing its behaviour. And this is why !copy returns `InputTensor`, not `AbstractTensor`.

See also: `!copy-force` never being ignored by compiler, and broadcasted axes will be padded.

Input:  Tensor[AbstractTensor]
Output: Tensor[AbstractTensor]"

  ;; FIXME: Operation with tensor permuted...
  ;;(when (tensor-permuted-p tensor)
  ;;  (return-from !copy (->contiguous tensor)))
  
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
	 (res (!move out tensor :force force)))
    ;; Extend flexible-p, because !copy is used to make a cache before using basic-function like !add
    (extend-states res tensor)))

;; Depcrecated ...
(defun !copy-force (tensor)
  "
## [function] !copy-force

```lisp
(!copy-force (tensor))
```

The function !copy-force returns a node which copies the given tensor forcibly while the function !copy sometimes ignored.

This function is also used to adjust memory alignment of tensor."

  (warn "!copy-force is subject to be deleted in the future release. use instead: (!copy tensor :force t)")
  
  (let* ((out (make-tensor (if (scalar-p tensor)
			       0
			       (shape tensor))
			   :dtype (dtype tensor)
			   :order (order tensor)))
	 (res (!move out tensor)))
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
	     ((self dout dx dy) ;; (viewed-tensor old)
	      (let* ((out-sub (tensor-view dy))
		     (inp-sub (slot-value self 'subscripts))
		     (res (apply
			   #'!view
			   (!move dx (apply #'!view dout inp-sub))
			   out-sub)))
		(values nil res))))


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
"
  (let ((out (apply #'cl-waffe2/vm.generic-tensor::view tensor subscripts))
	(broadcast-reverser
	  (loop for s in subscripts
		if (and (listp s)
			(eql (car s) :broadcast))
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

(define-impl (ReshapeTensorNode :device t)
	     :save-for-backward (t) ;; =T is necessary not to delete MoveTensorNode.
	     :forward ((self x y)
		       (declare (ignore y))
		       `(progn
			  ;;(setf (tensor-vec ,y) (tensor-vec ,x))
			  ,x)))

;; ===============================================================
;; Reshaping APIs
;; ===============================================================

(defun parse-reshape-args (before-shape after-shape)
  "check after-shape is consisted of positive fixnum.
shapes can contain t at once, this function also infers t."

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

(declaim (ftype (function (AbstractTensor &rest (and (not null) (or boolean fixnum))) AbstractTensor) !reshape))
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

"
  (declare (type AbstractTensor tensor))
  
  (let* ((shapes (parse-reshape-args (shape tensor) shapes))
	 (result (make-input shapes nil
			     :dtype (dtype tensor)
			     :order (order tensor))))

    
    (assert (= (apply #'* (shape tensor))
	       (apply #'* shapes))
	    nil
	    "Reshaping failed because total size doesn't match.")
    ;; (!view tensor `(2 4) `(2 4)) -> Copy
    ;; (!view tensor  0 t t t)
    (let ((result
	    (if (tensor-projected-p tensor)
		(forward (ReshapeTensorNode (shape tensor) shapes) (!copy tensor) result)
		(forward (ReshapeTensorNode (shape tensor) shapes) tensor result))))
      result)))

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

(declaim (ftype (function (AbstractTensor fixnum) AbstractTensor) !rankup))
(defun !rankup (tensor ntimes)
  "
## [function] !rankup

```lisp
(!rankup tensor ntimes)
```

The function !rankup appends/reduces 1 into the given tensor's shape for ntimes.

1. If `ntimes` > 0, appends 1

2. If `ntimes` < 0, reduces 1, if the axis=1, otherwise returns error."
  (declare (type AbstractTensor tensor)
	   (type fixnum ntimes))
  (let ((shape (copy-list (shape tensor))))
    (if (< ntimes 0)
	(loop for i fixnum upfrom 0 below (abs ntimes)
	      do (if (= (car shape) 1)
		     (pop shape)
		     (error "!rankup failed because it encountered a dimension which is not the equivalent to 1.")))
	(loop for i fixnum upfrom 0 below ntimes
	      do (push 1 shape)))
    ;; TODO: view broadcast
    (apply #'!reshape tensor shape)))


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

(defnode (ProceedNode (myself &key (measure-time nil) (compile-mode :default))
	  :where (A[~] -> A[~])
	  :slots ((measure-time   :initarg :measure-time :reader measure-time-p)
		  (compiled-model :initform nil :accessor proceed-compiled-model)
		  (compile-mode   :initarg :compile-mode :reader compile-mode)
		  (result         :accessor proceed-result))
	  :documentation "ProceedNode is a special node which takes all the previous computation node before tensor."))

(define-impl (ProceedNode :device t)
	     :save-for-backward (nil)
	     :forward ((self x)
		       (let ((compiled-model (or
					      (proceed-compiled-model self)
					      (build x :compile-mode (compile-mode self)))))
			 ;; Compiled-Composite
			 (setf (proceed-compiled-model self) compiled-model)
			 
			 (if (measure-time-p self)
			     (progn
			       ;; TODO
			       ;; Display Both: First-Time-Call/Second-Time-Call
			       (format t "Proceed-Time: With allocation time:~%")
			       (time (forward compiled-model))
			       (format t "Proceed-Time: Without allocation time:~%")
			       (setf (proceed-result self) (time (forward compiled-model))))
			     (setf (proceed-result self) (forward compiled-model)))
			 ;; Tell cl-waffe2 VM the returned value's type
			 (setf (out-scalar-p self) (scalar-p (proceed-result self)))

			 ;; The result is returned.
			 `(progn
			    ;; Tell top compiling funtion what composite to use for the compiled-function
			    (cl-waffe2/vm.generic-tensor::declare-compiled-composite ,compiled-model)
			    
			    ,x)))
	     :backward ((self dout dx)
			(declare (ignore dx))
			(let ((compiled-model (proceed-compiled-model self)))
			  (values
			   (with-instant-kernel dout
			     `(and
			       ,(if (measure-time-p self)
				    `(progn
				       (format t "Proceed-Time: Backward Time")
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
  (let* ((node (ProceedNode :measure-time measure-time :compile-mode compile-mode))
	 ;; Previous Node is already compiled, so detach tensor from nodes.
	 (out (forward node tensor)))
    
    ;; Cut off previous backwards
    (setf (tensor-backward tensor) nil)
    (setf (cl-waffe2/vm.generic-tensor::detach-p tensor) t)

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

;; ===============================================================
;; Broadcast APIs
;; ===============================================================

(defnode (Flexible-Rank-Node (myself)
	  :where (A[~] -> A[~])
	  :backward ((self dout dx)
		     (declare (ignore dx))
		     (values (!flexible dout)))))

(define-impl (Flexible-Rank-Node :device t) :forward ((self x) `(progn ,x)))

(defun !flexible (tensor)
  "
## [function] !flexible

```
(!flexible tensor)
```

The function !flexible returns a node which adds 1 (which is broadcastable) to the head of the shape of tensor.

That is:

```
Tensor = (10 10) -> [!flexible] -> Tensor' = (1 ... 1 10 10)
                                                 ^ <1 x N>
```

Note that added axes could be broadcasted automatically when the operation called with multiple arguments."
  (let ((out (forward (Flexible-Rank-Node) tensor)))
    (setf (tensor-flexible-p out) t)
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


(define-and-impl-node (Permute-Node (self before after permute-old)
		       :slots ((permute-old :initform nil :initarg :permute-old :reader permute-old))
		       :where (Old[before] New[after] -> New[after])
		       :forward ((self a out)
				 (declare (ignore a))
				 `(progn ,out))
		       :backward ((self dout a out)
				  (declare (ignore a out))
				  (values
				   (apply #'!permute dout (permute-old self))
				   nil)))
  (setf (ignore-shape-error self) t))

(defun list-diff (lista listb)
  (loop for l1 in lista
	for l2 in listb
	collect (= l1 l2)))

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

"
  ;; If only the last two axes are subject to swapped.
  ;; Return a special node LazyTranspose instead.
  (let* ((new-tensor (apply #'permute* tensor orders))
	 (diff       (list-diff (cl-waffe2/vm.generic-tensor::tensor-permute-order tensor)
				(cl-waffe2/vm.generic-tensor::tensor-permute-order new-tensor)))
	 (lazy-p (and (every #'(lambda (x) x) (butlast diff 2))
		      (every #'null           (last    diff 2))))
	 (out  (forward (Permute-Node
			 (shape tensor)
			 (shape new-tensor)
			 (cl-waffe2/vm.generic-tensor::tensor-permute-order tensor))
			tensor
			new-tensor)))
    ;; The case when (T NIL NIL) (T T NIL NIL) (NIL NIL) ... subject to lazy-transpose
    
    ;; judge: the diff is last two?
    (if lazy-p
	(call (LazyTransposeNode) out)
	out)))

(with-export ->contiguous
  (defun ->contiguous (tensor)
    "
## [function] ->contiguous

"
    (if (or (tensor-projected-p tensor)
	    (tensor-permuted-p tensor))
	;; ScalarTensor never becomes projected array
	(let* ((contiguous-place (make-input (shape tensor) nil
					     :dtype (dtype tensor)
					     :order (order tensor))))
	  (!move contiguous-place tensor :force t))
	tensor)))


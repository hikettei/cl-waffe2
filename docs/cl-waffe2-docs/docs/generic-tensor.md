
# AbstractTensor

## Working with AbstractTensor

## [class] AbstractTensor

AbstractTensor is a CLOS class that Wraps existing data structures such as matrices in an abstract class in automatic differential programming using cl-waffe2, and further adds information about computation nodes, gradients, etc.

Tensors can be created by the `make-tensor` function.

```lisp
(make-tensor `(3 3))
```

Plus, InputTensors (lazy-evaluated tensors), which is used to delay allocation timing, to use dynamic shaping, and to store the result, can be created by the `make-input` function.

```lisp
(make-input `(3 3) :A) ;; Set :A=nil to register as a temporary space.
```

As an applied use, users can create new `AbstractTensor` that inherit from AbstractTensor. In addition, inheriting existing AbstractTensors (e.g.: `LispTensor` for CL Standard Array) allows reusing descriptions such as allocations.

```lisp
(defclass MyOriginalTensor (AbstractTensor) nil)
(defclass MyCPUTensor      (LispTensor) nil)
```

Declare the priority of the device to be used with the with-devices macro.

```lisp
;; Higher <-> Lower
(with-devices (MyCPUTensor MyOriginalTensor CPUTensor)
    (make-tensor `(10 10)))
```

All available devices can be accessed with the `(show-backends)` function, and they can only be used as devices together if they are shown to have an inheritance relationship.

If a completely new Tensor is defined from AbstractTensor, cl-waffe2 can handle it completely in a fast form by writing the following additional information.

- Allocator: `initialize-instance :before method`

- Storage Accessor: `vref` and `(setf vref)` method

- Finalizer: `tensor-finalizer` method

- (Optional) Backend State: `current-backend-state` method

- (Optional) a `cl-waffe2/vm:defpath` macro to enable device-specific optimization.

This is the simplest case of `MyTensor` which works on CL Standard Array.

```lisp
(defclass MyTensor (AbstractTensor) nil)

;; Allocators satisfy the following properties
;; 1. When facet is not `:exist`, do nothing.
;; 2. If `vec` is specified as an argument, use this, and do not allocate any tensors.
;; 3. Otherwise, allocate the tensor with:
;;     1. Dtype -> :dtype
;;     2. Size  -> :shape (must be 1D on the memory)
;;     3. initial-element -> :initial-element
(defmethod initialize-instance :before ((tensor MyTensor)
					&rest initargs
					&key &allow-other-keys)
  (let* ((shape (getf initargs :shape))
	 (dtype (dtype->lisp-type (getf initargs :dtype)))
	 (vec   (getf initargs :vec))
	 (facet (getf initargs :facet))
	 (initial-element (coerce (or (getf initargs :initial-element) 0) dtype)))
    (when (eql facet :exist)
      (if vec
	  (setf (tensor-vec tensor) vec)
	  (setf (tensor-vec tensor)
		(make-array
		 (apply #'* shape)
		 :element-type dtype
		 :initial-element initial-element))))))

;; vref reads the index th element of storage vec, this is must be a setfable.
;; Leave the annoying and complicated stride/offset computations to cl-waffe2!
(defmethod vref ((tensor MyTensor) index)
  (declare (type fixnum index))
  (aref (tensor-vec tensor) index))

(defmethod (setf vref) (new-value (tensor MyTensor) index)
  (declare (type fixnum index))
  (setf (aref (tensor-vec tensor) index) new-value))


;; The method should return a lambda function, if its storage vector isn't gc-reachable.
;; Finalizers are called when quitting (with-memory-pool ...) macro.
(defmethod tensor-finalizer ((tensor MyTensor))
    ;; Returning a dummy finalizer
    #'(lambda ()))

;; The function (show-backends) will display all devices and their information
;; If you want to put something, override this method and return a string.
(defmethod current-backend-state ((backend-name (eql 'MyTensor)))
  "Hello This is an demo")

;; For FusionOp and defpath macro usage, see the :cl-waffe2/vm docs.
```

`MyTensor` is now recognised as a usable device, so operations can be defined using the define-impl and define-impl-op macros.


### [function] shape

`(shape tensor)` returns a visible shape of the given tensor.

### [function] dims

`(dims tensor)` returns a rank of the given tensor.

### [function] total

`(total tensor)` returns the number of total visible elements of the giventensor.

### [slot] orig-shape (List)

stores the shape of storage vec.

### [accessor] initial-offset (fixnum)

stores the offset of the tensor. In default, set to 0. Shape testing, for example, does not work, so use with caution.

`(tensor-initial-offset tensor)`

### [slot] stride (list)

`(tensor-stride tensor)` stores the stride of tensor.

### [slot] visible-shape (list)

`(shape tensor)`

### [slot] view (list)

Returns a list of ViewInstruction, created by the function `(view tensor ...)` or `(!view tensor ...)` to create a backward.

`(tensor-view tensor)`

### [slot] projected-p (boolean)

Set t if `(apply #'* orig-shape) == (apply #'* visible-shape)` otherwise set nil.

If t, the tensor is created by `!view` or `view` functions.

### [slot] scalar-p

Set t if the tensor should be represented as a scalar. In cl-waffe2, it's not a pretty thing but scalars are represented as a `(apply #'* shape)=1` tensors. ranks are anything but for the most case, returns 1.

### [slot] detach-p

Set T to detach the tensor at a certain position.

### [slot] state

(tensor-state tensor) stores `StateContainer`.

### [slot] variables

`(tensor-variables tensor)` stores the previous variables if the tensor is created by any operation.

### [slot] tensor-id (symbol)

Indicates where the Tensor is stored, (e.g. in a virtual machine). In-place operations inherit tensor-id from variables called with, and should not be used for topological sorting.

### [slot] tensor-iid (symbol)

It holds an ID that is guaranteed to be absolutely unique to the processing system generated by gensym. Used for topological sorting.

### [slot] grad (AbstractTensor)

If the tensor is created by (parameter ...) or with `:requires-grad=t`, `(grad tensor)` will return a gradient.

### [slot] backward (AbstractNode)

`(tensor-backward tensor)` returns a abstractnode if the tensor is created by any operation.

### [slot] requires-grad (Boolean)

Set T to hold the gradients.

### [slot] ancestor-param-p (Boolean)

Set T if compilers can reach any tensors with `:requires-grad=t`, by tracing the tensor.

### [slot] flexible-p (Fixnum or Null)

Indicates the position of broadcastable axis.

### [slot] facet (keyword)

AbstractTensors in cl-waffe2 has a two state: `ExistTensor` and `InputTensor`. `ExistTensor` is a just tensor with allocated storage vec, made by make-tensor function. On the other hand InputTensor is a lazy-evaluated tensor, allocation won't be done until it is needed.

:exist to ExitTensor, :input to InputTensor.

### [method] mref

`(mref tensor &rest subscripts)` will reads a cetrain position of storage vec. This is setfable. In terms of performance, it is much faster way to edit a storage vec that using `(change-facet)` function and convert into other forms.

### Hooking Optimizers and Optimizing Parameters

(TODO)



## [function] hook-optimizer!

```lisp
(hook-optimizer! tensor optimizer)
```

Hooks the optimizer to the tensor.

### Inputs

tensor[AbstractTensor]

optimizer[AbstractOptimizer]


## [function] call-optimizer!

```lisp
(call-optimizer! tensor)
```

Reading the `(grad tensor)`, the function invokes the optimizer hooked to the tensor.

## [function] reset-grad!

Resets the gradient of the tensor with zero with `retain-grad=t`.


## [function] tensor-vec

```lisp
(tensor-vec tensor)
```

If the given tensor is a ExistTensor, returns its storage vec.

If the given tensor is a InputTensor, allocates the area for tensor and return its storage vec.

This function is setfable and inlined.

## [function] make-tensor

```
(make-tensor shape-or-scalar
	       &key
		  (requires-grad nil)
		  (dtype *default-dtype*)
		  (view nil)
		  (order *default-order*)
		  (initial-element nil)
                  (device nil))
```

Created a new ExistTensor of a device of `(car *using-backend*)`.

### Inputs

1. `shape-or-scalar`[Anything] If set to list, creates a new matrix. Otherwise (e.g.: set to fixnum), creates a ScalarTensor. In that case, cl-waffe2 uses the highest priority device from `*using-backends*` parameter that inherits from the `ScalarTensor` class.

2. `requires-grad`[Boolean] Set t to holds a gradients. `(parameter tensor)` will also do the same work. Under `(with-no-grad ...)` macro. This is set to nil forcibly.

3. `dtype`[keyword] Set keyword indicating a type of elements.

4. `order`[keyword] set keyword indicating the order of elments from `:column` or `:row`. in default set to `:column`.

5. `initial-element`[Anything] Set anything which you want to set as a initial element.

6. `device[symbol or null]` If set to symbol, the function returns with making a tensor of device.

### Example

```lisp
(make-tensor `(10 10) :initial-element 1.0)

{CPUTENSOR[float] :shape (10 10)  
  ((1.0 1.0 1.0 ~ 1.0 1.0 1.0)           
   (1.0 1.0 1.0 ~ 1.0 1.0 1.0)   
        ...
   (1.0 1.0 1.0 ~ 1.0 1.0 1.0)
   (1.0 1.0 1.0 ~ 1.0 1.0 1.0))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## [function] make-input

```lisp
(make-input shape named &key (created-from nil) (scalar-p nil) (dtype *default-dtype*) (order *default-order*))
```

Creates a new InputTensor. The allocation won't be done until the function `(tensor-vec tensor)` is called. In cl-waffe2, InputTensors can be applied for various things, for example, tracing the structure of computation node, used as a temporary tensor which can be pruned later by a compiler, as an argument of the computation node compiled by the `build` function. 

### Inputs

`Shape` [list] Set the shape of tensor. You can also use symbols if shapes can be changed later. The function `set-input` will update all symbols declared in the computation node, and accordingly, strides/shapes etc... will be also updated to minimise compiling-time overhead (use `build` and `forward` to do this). ScalarTensors aren't created by setting it=`<<Something but not a list>>`. Instead, set `scalar-p=t`.

`Named` [keyword or null] Indicates the name of tensor. If set to keyword, This means the name of the argument when compiled into a function, which can be changed later. If set to nil, the name is filled with `gensym` indicating the index in the memory-pool.

`scalar-p` [boolean] Set t to create a scalar.

`dtype` [keyword] Set dtype.

`order` [keyword] Set order.

`create-from[nil or AbstractTensor]` The returned InputTensor will extend Permutions/Strides and so on from `create-from` if any.

### Example

```lisp
(make-input `(a 10) :train-x)

{CPUTENSOR[float] :shape (A 10) :named :TRAIN-X 
    <<Not allocated: size=(A 10)>>
  :facet :input
  :requires-grad NIL
  :backward NIL}
```

## Manipulating Gradients

## [parameter] `*no-grad*`

Ensures that back-propagation is not invoked inside the scope for which this parameter is set to T, with the following effects:

- Save For Backward is forcibly ignored.

- Computational nodes for back propagation are not compiled.

In default, set to nil. See also the `with-no-grad` macro to explict this state.

## [macro] with-no-grad

```lisp
(with-no-grad &body body)
```

Set T to `*no-grad*` during the execution of body.

## [function] parameter

```
(parameter tensor)
```

Creates a new tensor with :requires-grad=t from the given tensor. If the tensor is remained to be computed, parameter will use the result from `proceed`.

### Example

```lisp
(parameter (randn `(3 3)))
```

## Building functions from AbstractTensor

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

###Example
**REPL:**
```lisp
> (setq out (!add (make-input `(a 10) :X) (make-input `(a 10) :Y)))
```
```
{CPUTENSOR[float] :shape (A 10) :id TID1830 
  :vec-state [maybe-not-computed]
    <<Not allocated: size=(A 10)>>
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>}
```

**REPL:**
```lisp
> (with-no-grad (build out :inputs `(:X :Y)))
```
```
<Compiled-Composite(allocated-p=NIL)
    forward     : forward(model X Y) -> CPUTENSOR{FLOAT}(A 10)
    backward    : nil
    memory-pool : one tensor(s)
                   L {4.0e-5+((A) x 4.0e-6)}MB
    inputs:
        X -> (A 10)
        Y -> (A 10)
>
```

## [method] set-input

```
(set-input (model Compiled-Composite) input-name actual-value)
```

Embodies an `InputTensor` in the model. All unembodied tensors in the model can be accessed by printing the model.

`input-name` could be a keyword indicating input-tensor, `actual-value` is a `AbstractTensor` whose facet = `:exist` (created by `make-tensor`).

## [method] get-input

```
(get-input (model Compiled-Composite) input-name)
```

Reading all variables in the computation node, the method get-input returns an corresponding `InputTensor` of model.

## Creating a ranked function with computing views

## [function] call-with-view

A principle operator to extend your functions to higher arrays.

```lisp
(call-with-view function tensors &key (at-least-dim 1) (force-order nil) (lparallel nil))
```

The function `call-with-view` generates a lisp code of `(loop for ...)` iteration for nd-arrays, which follows the optimal route, is parallelized, and later composable. Since generating an optimal `for(int i=0;i<size;i++){...}` route according to the given rank of tensors is one of the main concerns of JIT Compiler for Deep Learning Framework, this function is usually combined with the forward definition of `define-impl` macro. It is later compiled to lambda functions and used as nodes in cl-waffe2 IR.

In the simplest case, `call-with-view` first deploys `(loop for...)` until the rank of given tensors reaches the given `at-least-dim`. After reaching `at-least-dim`, the function places the result of calling the given `function`.

```lisp
(call-with-view
      #'(lambda (x-view)
	   `(+ 1 1))
       (list (randn `(100 100 100)))
       :at-least-dim 2)

;; will return:

(CL-WAFFE2/VM.GENERIC-TENSOR::LET*-IGNORABLE ((#:G312057 0))
  (LOCALLY
   (DECLARE (TYPE FIXNUM #:G312057))
   (CL-WAFFE2/VM.GENERIC-TENSOR::LET*-IGNORABLE ((#:G312058 #:G312057))
     (LOCALLY
      (DECLARE (TYPE FIXNUM #:G312058))
      (LET* ((#:G312059 (NTH 0 (LIST 10000 100 1)))
             (#:G25 100)
             (#:G25
              (CL-WAFFE2/VM.GENERIC-TENSOR::READ-ADJUSTABLE-SYMBOL #:G25)))
        (INCF #:G312058 (CL-WAFFE2/VM.GENERIC-TENSOR::%* 0 #:G312059))
        (LOOP CL-WAFFE2/VM.GENERIC-TENSOR::FOR #:G312060 FIXNUM CL-WAFFE2/VM.GENERIC-TENSOR::UPFROM 0 CL-WAFFE2/VM.GENERIC-TENSOR::BELOW #:G25
              DO (PROGN
                  (CL-WAFFE2/VM.GENERIC-TENSOR::LET*-IGNORABLE ((#:G312061
                                                                 #:G312058))
                    (LOCALLY
                     (DECLARE (TYPE FIXNUM #:G312061))
                     (LET ((#:G312062 (THE FIXNUM (NTH 1 (LIST 10000 100 1)))))
                       (INCF #:G312061
                             (CL-WAFFE2/VM.GENERIC-TENSOR::%* 0 #:G312062))
                       (+ 1 1)))))
              UNLESS (= #:G312060 (1- #:G25))
              DO (PROGN
                  (INCF (THE FIXNUM #:G312058) (THE FIXNUM #:G312059)))))))))
```

Here, the number of tensors corresponds with the number of arguments `function` receive. Usually, the function receives information on the view of the tensor at the corresponding position: `(size-of x-view)` to get the number of iteration, `(stride-of x-view)` to get the number of increment, and, `(offset-of x-view)` to get the offset of tensor. (Sometimes they return s-expression because the shapes of tensors are not necessary number, but symbols.)

`function [function]` should return a list which corresponds with invoking user-defined operation given views.

`tensors[a list of abstracttensor]` tensors to be called with.

`at-least-dim [fixnum]` `at-least-dim is minimum rank value required by the operation. set 1 to define `element-wise` operation, set 2 to define `gemm` for example.

`force-order[boolean]` On some conditions, `call-with-view` shuffles the order of ranks, or flattens given tensors (e.g.: `100x100` tensors is the equivalent to just `10000x1` tensor on the memory). If you want to disable this behaviour, set `force-order`=t.

`lparallel[boolean]` Set t to use lparallel. This should be denoted that under lparallel execution, the parameter `cl-waffe2/threads:*under-multi-thread*` becomes t. Use this parameter for the lowest rank operation to decide whether to parallelise.

Return: `Expanded Lisp Codes`

Note that `call-with-view` should be used at once or zero in the one `define-impl` forward. If you need twice times to call it, the general definition of `AbstractNode` should be split.

See also: `with-ranked-loop` to the more elegant wrapping macro.

## [macro] with-ranked-loop


```lisp
(with-ranked-loop (((op-function &rest variables)
                    &key
                       (kernel-size 1)
                       (shuffle-rank t)
                       (lparallel nil))
                    &body body))
```

Just an alias of `call-with-view` with this form:

```lisp
`(,@(call-with-view op-function variables :at-least-dim kernel-size :force-order (not shuffle-rank) :lparallel lparallel :fuse fuse)
  ,@body)
```

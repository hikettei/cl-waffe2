
# AbstractTensor

## [class] AbstractTensor

[class] `AbstractTensor`


AbstractTensor is a primal class for all devices. Each devices (e.g.: `ScalarTensor` `LispTensor` `CPUTensor` etc...) is a subclass of this.

The class provides the fundamental and necessary features for tensors.

1. Lazy-Evaluated and Multi-Dimensional APIs, stride computations.

2. `View APIs` multi-dimensional offsets

3. To construct backward, AbstractTensor records variables called with.
'
4. `vec` container.

5. an space for saving gradients, copies for backward.

6. Lazy-Evaluated Shapings

7. Trace Informations for JIT to create well-optimized computation node.

### Creating a new backend.

Users can create a new backend by extending this abstract class.

```lisp
(defclass MyBackend (AbstractNode) nil)
```

To use the `MyBackend` as a tensor, users also has to override these methods:

1. `initialize-instance` ... An allocator for tensor's vec.

2. `vref` `(setf vref)` ... an generic function to access/write tensor's vec.

```lisp
;; TODO: Establish a common API for initargs
(defmethod initialize-instance :before ((tensor MyBackend)
					&rest initargs
					&key &allow-other-keys)
  ;; if projected-p -> alloc new vec
  (let* ((shape (getf initargs :shape))
 	 (dtype (dtype->lisp-type (getf initargs :dtype)))
	 (vec   (getf initargs :vec))
	 (facet (getf initargs :facet))
	 (initial-element (coerce (or (getf initargs :initial-element) 0) dtype)))
    (when (eql facet :exist)
      (if vec
	  (setf (tensor-vec tensor) vec)
	  (setf (tensor-vec tensor) ;; vec can be anything.
		(make-array
		 (apply #'* shape)
		 :element-type dtype
		 :initial-element initial-element))))))
```

```lisp
(defmethod vref ((tensor MyBackend) index)
  (aref (tensor-vec tensor) index))

(defmethod (setf vref) (new-value (tensor MyBackend) index)
  (setf (aref (tensor-vec tensor) index) new-value))
```

Now, the name `MyBackend` is available as a brand-new cl-waffe2 backend!

Users can define a new implementation following `(define-impl (Name :device MyBackend) ...)`

(See the examples to understand how this could be achieved at ./source/backends/lisp/tensor.lisp. or ./source/backends/cpu.)

### [function] shape

Returns a visible shape of tensor

### [function] dims

Returns the number of axes of tensor

### [function] total

Returns the number of total visible elements in tensor.

### [slot] orig-shape (List)

the original shape of `vec`. `(apply #'* orig-shape)` must correspond with the number of total elements of `vec`.

### [slot] stride (list)

An stride of tensor, can be chosen from `:column` `:row`.

This slot can be accessed by `(tensor-stride object)`.

### [slot] visible-shape (list)

An shape of visible-area of tensor, visible-area is that an viewed size of tensor.

Can be accessed by `(shape object)`

### [slot] view (list)

An list of multidimensional offsets, view.

Can be accessed by `(tensor-view object)`

### [slot] projected-p (boolean)

Set t if `(apply #'* orig-shape) == (apply #'* visible-shape)` otherwise set nil.

If t, the tensor is produced by `!view` or `view` functions.

### [slot] scalar-p

If t, the tensor is regarded as a Scalar.

### [slot] detach-p

If t, JIT compilers stop tracing at the tensor.

### [slot] state

Stores a corresponding `StateContainer`.

### [slot] variables

`(tensor-variables object)`

Records variables called with the tensor.

### [slot] tensor-id (symbol)

Corresponding variable name that used in JIT compiler.

### [slot] grad (AbstractTensor)

If the tensor is a parameter, (i.e.: requires-grad t) and backward propagation has called, the gradients has set to this slot.

Reader: `(grad object)`.  Writer: `(set-grad object value)`

### [slot] backward (AbstractNode)

the node called with.

### [slot] requires-grad (Boolean)

If t, the tensor become a `parameter` that gradients are saved.

### [slot] ancestor-param-p (Boolean)

If t, the tensor has created by `parameter` or tensors whose ancestor-param-p=t.

### [slot] flexible-p (Boolean)

Set fixnum to add broadcastable axis.

### [slot] facet (keyword)

Tensors has a two state:

1. :input

2. :exist

`:exist` tensor is a just normal state, which `vec` is already allocated.

`:input` tensor is a lazy-evaluated tensor, which allocation will be done until they're really needed. (often used as a cache, or training data.)
...


## [function] tensor-vec

`(tensor-vec tensor)`

Reading the `vec` of tensor.  Not until tensor-vec is called, the new area isn't allocated.
## [function] mref

`(mref tensor &rest subscripts)`

The function mref is only used to print/initialize tensors, accessing the index of subscripts **with** considering views..

If you cares about performance, dont' use `mref`, but `!view`.

This function is setfable.
## [generic] vref

`(vref tensor index)`

`vref` is a generic-function to access the `vec` slot of specific backends tensor, and returns `index`th element on `vec` slot **without** considering views.

If you added a new backend with having different ptr-type (can't be accessed by aref), override this method and `(setf vref)`.

### Example

```lisp
(defmethod vref ((tensor YourBackend) index)
    (aref (tensor-vec tensor) index))

(defmethod (setf vref) (new-value (tensor YourBackend) index)
    (setf (aref (tensor-vec tensor) index) new-value))
```

## An form of tensors in cl-waffe2
There's a two type of tensors in cl-waffe2: `InputTensor` and `ExistTensor`, each state is called `facet` and the keyword `:input` `:exist` is dispatched respectively.
### ExistTensor
`ExistTensor` means a tensor with its vec **allocated** in the memory, that is, the same tensor as tensors you got when create a new tensor in `Numpy`, `PyTorch` or something.

`ExistTensor` can be created by the function `make-tensor`.


### InputTensor
On the other hand, `InputTensor` is a tensor with its vec **unallocated** in the memory, in other words, this can be a `Lazy-Evaluated Tensor`.

`InputTensor` is created by the function `make-input`, and its shape can include a symbol.

In the network, `InputTensor` plays a role in being caches in the operation, or being a tensor that one may want to change its content later. (e.g.: training data).


## [function] make-tensor

```
(make-tensor shape-or-scalar
		   &key
		      (requires-grad nil)
		      (dtype *default-dtype*)
		      (view nil)
		      (order *default-order*)
		      (initial-element))
```

Refering a first-priority of  *using-backends* (i.e.: `car` of `*using-backends*`) to know what device to use, the function `make-tensor` creates and allocate a new matrix instantly.

### Input

1. `shape-or-scalar (Any)` set list (consisted of fixnum) here to create a matrix, otherwise the ScalarTensor is forcibly created.

2. `requires-grad` (Boolean) Set t to create gradient. (e.g.: the tensor is needed to be optimized.)

3. `dtype` (keyword) Set dtype you wanna use. See also: (Dtype API)

4. `order` (member :column :row)

5. `initial-element` (Optional)

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

Referring a first-priority of `*using-backend*` (i.e.: car part), the function make-input creates a InputTensor.

In contrast to `make-tensor`, allocation of `vec` is lazy-evaluated, and `shape` can include symbols. (Lazy-Evaluated Shape).

For example, whichever `(make-input (list 256 256 256 ... 256 256 256) nil)` or `(make-input (list 256) nil)` is called, the memory-usage is the same until `(tensor-vec tensor)` is called but the moment `(tensor-vec tensor)` is called, the first one would cause `CUDA OUT OF MEMORY` or something :(.

### Inputs

`Shape` [list] consisted of fixnum or symbol. (e.g.: `(a 10)` is OK for make-input.)

`Named` [keyword] the name of input. If nil, the tensor is regarded as just cache. If you want to change the content of inputs later (e.g.: training data), set an appropriate name to `InputTensor` (e.g.: `:training-data` `:train-x`).

`scalar-p` [boolean] set t is the input is scalar.

`dtype` [keyword] as it is.

`order` [keyword] an member of :column :row

`create-from[nil or AbstractTensor]` If you want to extend permute state/stride information, fill it.

### Example

```lisp
(make-input `(a 10) :train-x)

{CPUTENSOR[float] :shape (A 10) :named TRAIN-X 
  <<Not-Embodied (A 10) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward NIL}
```
The InputTensor named with a keyword is called `not-embodied tensor`, and can be changed its `vec` with `embody-input`
## [class] Compiled-Composite

Compiled-Composite is a `callable` CLOS class, and holds compiled forward/backward function of all the computation node to all the endpoints from the top of the models' neural network. Also, this class holds information of all variables used in the node.

It is NOT possible to construct a computation node after Compiled-Composite, If you need this, try consider using the function `cl-waffe2/base-impl:proceed`.

The class will appear in your project with calling the function `build`, set the toplevel node (e.g.: the result of criterion when the task is optimizing.) to the first argument. cl-waffe2 compiler will instantly construct an lambda function of forward/backward, which is invoked by calling `(forward compiled-composite)` or `(backward compiled-composite)` method.

See also: `build` `set-input` `get-input`.

### Examples

(TODO)


## [function] build

```lisp
(build toplevel
	      &key
		(construct-backward? (not *no-grad*))
		(compile-mode :fastest))
```

Receiving the toplevel node in the neural network, the function `build` constructs a optimal forward/backward function, returning `Compiled-Composite`.

### The constraints of toplevel tensor.

The shape of topleve mustn't include a `symbol`.

For example, this cl-waffe2 operation is invaild. because the function `(!sin x)` still returns `(A B)` tensor.

```lisp
(build (!sin (make-input `(A B) :Input)))
```

In order to build this operation correctly, calling `criterion` (intrinsically, `!sum` or `!mean`) is a vaild option for neural network tasks.

```lisp
(build (!sum (!sin (make-input `(A B) :input)))) ;; Passes Correctly!
```

After working with adjustable shape tensor, don't forget to embody the InputTensor!

```lisp
(let ((compiled-model (build (!sum (!sin (make-input `(A B) :input))))))
    (set-input compiled-model :input (randn `(10 10)))
    (forward compiled-model))
```

### Inputs

`toplevel [AbstractTensor]` the end of nodes. for neural network tasks, this should be scalartensor or tensors with total elements is 1, but since cl-waffe2 is intended to be applied other tasks, cl-waffe2 never produce warning while other frameworks like PyTorch will return error if `<<(10 10)Tensor>>.backward()` is invoked for example.

`construct-backward?` [boolean] If t, the backward construction won't be done.

`compile-mode`[compile-mode-t] an keyword to indicate compiling option.

###Example
**REPL:**
```lisp
> (setq out (!add (randn `(10 10)) (make-input `(a 10) :X)))
```
```
{CPUTENSOR[float] :shape (10 10) :named ChainTMP1649 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (10 10) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>}
```

**REPL:**
```lisp
> (multiple-value-list (build out))
```
```
(<Compiled-Composite
    forward:  #<FUNCTION (LAMBDA ()
                           :IN
                           "/Users/hikettei/.cache/common-lisp/sbcl-2.3.3-macosx-x64/Users/hikettei/Desktop/cl-waffe-workspace/progs/develop/cl-waffe2/docs/apis/generic-tensor.fasl") {53AA916B}>
    backward: #<FUNCTION (LAMBDA ()
                           :IN
                           "/Users/hikettei/.cache/common-lisp/sbcl-2.3.3-macosx-x64/Users/hikettei/Desktop/cl-waffe-workspace/progs/develop/cl-waffe2/docs/apis/generic-tensor.fasl") {53AAA7CB}>

+= [Tensors in the computation node] =======+

Subscripts:
     [A -> ?, max=?]


Variables:
 NAMES |   SIZE  | 
––––––––––––––––––
   X   |  (A 10) | 


 - The number of tmp variables : 6
 - The number of parameters    : 0
+========================================+
>)
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


## [parameter] `*no-grad*`
[parameter] `*no-grad*`

If t, no gradients are made for backwards.
## [macro] with-no-grad

```lisp
(with-no-grad &body body)
```

Under the `body` execution, the macro sets `*no-grad*` = `t`, that is, the built nodes are regarded as: no gradients are made for backwards.



## [function] parameter

```
(parameter tensor)
```

The function parameter computes all the previous nodes of the given tensor if any, returning the new tensor with `requires-grad=t`.

### Example

```lisp
(parameter (randn `(3 3)))
```


## [function] call-with-view

```lisp
(call-with-view function tensors &key (at-least-dim 1))
```

`call-with-view` is a general-purpose interface to iterate multi-dimensional tensor with considering offsets.

(TODO: Example/Documents)

`function` [lambda] an lambda function which receives `variable1.view variable2.view ...` as arguments, returning an list being compiled.

`tensors` [list of abstracttensor] tensors to be called with.
`at-least-dim` [fixnum] ... kernel-size

See also:

`size-of`
`stride-of`
`offset-of`
NILNILNIL
## [function] shape-equal

a=1, b=k => T
a=1, b=2 => NIL

...Returns subscript-t if view is Subscript otherwise returns a view
## Compiling Options
TODO
## Dtypes
TODO
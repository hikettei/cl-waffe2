
# AbstractTensor

## AbstractTensor

[class] `AbstractTensor`


AbstractTensor is a primal class for all devices. Each devices (e.g.: `ScalarTensor` `LispTensor` `CPUTensor` etc...) is a subclass of this.

The class provides the fundamental and necessary features for tensors.

1. Lazy-Evaluated and Multi-Dimensional APIs, stride computations.

2. `View APIs` multi-dimensional offsets

3. To construct backward, AbstractTensor records variables called with.

4. `vec` container.

5. an space for saving gradients, copies for backward.

6. Lazy-Evaluated Shapings

7. Trace Informations for JIT to create well-optimized computation node.

### Create a new backend.

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

If t, the tensor is `broadcastable`

### [slot] facet (keyword)

Tensors has a two state:

1. :input

2. :exist

`:exist` tensor is a just normal state, which `vec` is already allocated.

`:input` tensor is a lazy-evaluated tensor, which allocation will be done until they're really needed. (often used as a cache, or training data.)
...


## make-tensor

## [function] make-tensor

```
(make-tensor shape-or-scalar
		   &key
		      (requires-grad nil)
		      (dtype *default-dtype*)
		      (vec  nil)
		      (view nil)
		      (order *default-order*)
		      (initial-element))
```

Refering a first-priority of  *using-backends* (i.e.: `car` of `*using-backends*`) to know what device to use, the function `make-tensor` creates and allocate a new matrix instantly.

### Input

1. `shape-or-scalar (Any)` set list (consisted of fixnum) here to create a matrix, otherwise the ScalarTensor is forcibly created.

2. `requires-grad` (Boolean) Set t to create gradient. (e.g.: the tensor is needed to be optimized.)

3. `dtype` (keyword) Set dtype you wanna use. See also: (Dtype API)

4. `vec` (Anything) If you wanna pass the make-instance to already-allocated matrix, use this parameter.

5. `order` (member :column :row)

6. `initial-element` (Optional)

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

## make-input

## [function] make-input

Referring a first-priority of *using-backend* (i.e.: car part), the function make-input creates a InputTensor.

In contrast to `make-tensor`, allocation of `vec` is lazy-evaluated, and `shape` can include symbols. (Lazy-Evaluated Shape).

For example, whichever `(make-input (list 256 256 256 256 256 256) nil)` or `(make-input (list 256) nil)` is called, the memory-usage is the same until `(tensor-vec tensor)` is called but the moment `(tensor-vec tensor)` is called, the first one would cause `CUDA OUT OF MEMORY` or something :(.

### Inputs

`Shape` [list] consisted of fixnum or symbol. (e.g.: `(a 10)` is a valid shape.)

`Named` [keyword] the name of input. If nil, the tensor is regarded as just cache. If you want to change the content of inputs later (e.g.: training data), set an appropriate name.

`scalar-p` [boolean] set t is the input is scalar.

`dtype` [keyword] as it is.

`order` [keyword] an member of :column :row

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
## embody-input
(embody-input variables :a tensor)
###Example
**REPL:**
```lisp
> (setq out (!add (randn `(10 10)) (make-input `(a 10) :x)))
```
```
{CPUTENSOR[float] :shape (10 10) :named ChainTMP35895 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (10 10) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>}
```

**REPL:**
```lisp
> (with-build (fw bw vars params) out
            (embody-input vars :x (randn `(10 10))) ;; :X = (randn `(10 10))
            (funcall fw))
```
```
{CPUTENSOR[float] :shape (10 10) :named ChainTMP35884 
  ((-2.1486177   1.4877725    -1.7822108   ~ 0.30888113   -3.668074    -1.4501324)                    
   (0.90827906   -3.6974688   -0.7262471   ~ 2.153652     0.7110309    1.2819712)   
                 ...
   (-2.6074939   0.04147309   -0.97653854  ~ 0.3843904    -0.20308924  -0.614793)
   (1.7244194    1.5219165    0.3820825    ~ -0.41161555  0.5861892    0.18113303))
  :facet :input
  :requires-grad NIL
  :backward NIL}
```

## build
Return:
    (values forward backward variables parameters)
###Example
**REPL:**
```lisp
> (setq out (!add (randn `(10 10)) (make-input `(a 10) :X)))
```
```
{CPUTENSOR[float] :shape (10 10) :named ChainTMP35924 
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
(#<FUNCTION (LAMBDA () :IN "/private/var/tmp/slimemU2Krr.fasl") {53A5B54B}>
 #<FUNCTION (LAMBDA () :IN "/private/var/tmp/slimemU2Krr.fasl") {53A8776B}>
 += [Computation Node Information] =======+

Subscripts:
     [A -> ?]


Variables
 NAMES |   SIZE  | 
––––––––––––––––––
   X   |  (A 10) | 


 - The number of tmp variables: 4
+========================================+
 #S(NODEPARAMETERS
    :PARAMETERS (omitted)
    :ntensors 3))
```

## tensor-vec

`(tensor-vec tensor)`

Accessing the pointer/array the tensor has. Not until tensor-vec is called, the new area isn't allocated.
## mref

`(mref tensor &rest subscripts)`

Read-only. Only used for printing the tensor.
Whether you cares about performance or not, this function shouldn't be used ignoring for printing tensors.
## vref

`(vref tensor index)`

vref is a generic-function to access tensor's vec.

Whether you cares about performance or not, this function shouldn't be used ignoring for printing tensors.

If you've created a new backend with having different ptr-type (can't be accessed by aref), only you have to do is to redefine vref.
## set-save-for-backward
NIL
## read-save-for-backward
NIL
## `*no-grad*`
[parameter] `*no-grad*`

If t, all operations don't create gradients.
## with-no-grad

## [macro] with-no-grad

```lisp
(with-no-grad &body body)
```

Set `*np-grad*` `t` under the `body` execution, no gradients are made for backward.

## parameter
The function parameter computes all the previous nodes of the given tensor, returning the new tensor with requires-grad=t.

Example:

```lisp
(parameter (randn `(3 3)))
```
## dtype->lisp-type
NIL
## call-with-view
NIL
## stride-of
NIL
## size-of
NIL
## offset-of
NIL
## shape-equal

## [function] shape-equal

a=1, b=k => T
a=1, b=2 => NIL

...
## force-list
Returns subscript-t if view is Subscript otherwise returns a view
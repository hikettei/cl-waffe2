
# [package] cl-waffe2
The `cl-waffe2` package provides utilities for a wide range needs: Object Convertion, Advance Network Construction, Trainer, and so on.
## [Tensor Facet] Converting AbstractTensor <-> Anything
If you're looking for the way to create an AbstractTensor from a Common Lisp array or manipulate an AbstractTensor as a Common Lisp array, this section is perfect for you. Here we provide a common APIs for the conversion between AbstractTensor and other matrix types. The most basic method is a `convert-tensor-facet` method and we're welcome to add a new method by users. Other functions are macros work by assigning a method according to the type of object and the direction. Plus, conversions are performed while sharing pointers as much as possible. If turned out to be they can't be shared, the with-facet macro forces a copy to be performed and pseudo-synchronises them.

## [generic] convert-tensor-facet

```lisp
(convert-tensor-facet from to)
```

Converts the given object (anything is ok; from=`AbstractTensor` `simple-array` etc as long as declared) into the direction indicated in `to`.

### Inputs

`From[Anything]` The object to be converted

`To[Symbol]` Indicates to where the object is converted

### Adding an extension

Welcome to define the addition of method by users. For example, `Fixnum -> AbstractTensor` convertion can be written like:

```lisp
(defmethod convert-tensor-facet ((from fixnum) (to (eql 'AbstractTensor)))
    (make-tensor from))

(print (change-facet 1 :direction 'AbstractTensor))

;;{SCALARTENSOR[float]   
;;    1.0
;;  :facet :exist
;;  :requires-grad NIL
;;  :backward NIL} 
```

If any object to AbstractTensor conversion is implemented, it is strongly recommended to add it to this method.

### Example

```lisp
(convert-tensor-facet (randn `(3 3)) 'simple-array)
```

See also: `convert-facet (more convenient API)`

## [function] change-facet

```lisp
(change-facet (array-from &key (direction 'array)))
```

By calling the `conver-tensor-facet` method, this function can change the facet of given `array-form` into the `direction`. (Just an alias of `(convert-tensor-facet array-from direction)`)

See also: `convert-tensor-facet`

### Standard Directions

We provide these symbols as a `direction` in standard.

- `array`: Any Object -> Common Lisp Standard ND Array

- `simple-array`: Any Object -> Common Lisp Simple-Array

- `AbstractTensor`: Any Object -> AbstractTensor. If couldn't determine the dtype, dtype of the first element of `array-from` is used instead.

## [function] ->tensor

Using the `convert-tensor-facet` method, converts the given object into AbstractTensor.

### Example

```lisp
(->tensor #2A((1 2 3) (4 5 6)))
```

## [macro] with-facet

```lisp
(with-facet (var (object-from &key (direction 'simple-array)) &body body))
```

By using the convert-tensor-facet` method, this macro changes the facet of `object-from` into the `direction`. If you want to apply any operations to `object-from` and ensure that modifications are applied to the `object-from`, set `sync`=t and moves element forcibly (only available when direction=`'abstracttensor`). This is useful when editing AbstractTensor or calling other libraries without making copies.

For a more explict workflow, see below:

```lisp
    [macro with-facet]
            ↓
[Binding var = (convert-tensor-facet object-from direction)] 
            ↓
      [Processing body]
            ↓
[If sync=t, (setf (tensor-vec object-from) (tensor-vec (convert-tensor-facet var 'AbstractTensor)))]
```

### Example

```lisp
(let ((a (randn `(3 3))))
    (with-facet (a* (a :direction 'simple-array))
        (print a*)
        (setf (aref a* 0) 10.0))
   a)

;; Operations called with simple-array a*, also effects on a.

#(0.92887694 -0.710253 1.2339028 -0.78008 1.6763965 0.93389416 -0.5691122
  1.6552123 -0.108502984) 
{CPUTENSOR[float] :shape (3 3)  
  ((10.0         -0.710253    1.2339028)
   (-0.78008     1.6763965    0.93389416)
   (-0.5691122   1.6552123    -0.108502984))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

See also: `with-facets`

## [macro] with-facets

Bundles several `with-facet` macro.

```lisp
(with-facets ((a ((randn `(3 3)) :direction 'array))
              (b ((randn `(3 3)) :direction 'array)))
    (print a)
    (print b))
#2A((-0.020553567 -0.016298171 -2.0616999)
    (0.68268335 0.33567926 -0.79862773)
    (1.7132819 0.8081283 0.47327513)) 
#2A((-0.9344233 0.3149136 -0.8516832)
    (0.17137305 -0.026806794 -0.8192844)
    (0.19916026 -0.5102597 1.1834184)) 
```

## Advanced Network Construction
Powerful macros in Common Lisp enabled me to provide an advanced APIs for make the construction of nodes more systematic, and elegant. Computational nodes that are lazy evaluated can be treated as pseudo-models, for example, even if they are created via functions. And, APIs in this section will make it easy to compose/compile several nodes.

## [function] asnode

```lisp
(asnode function &rest arguments)
```

Wraps the given `function` which excepted to create computation nodes with the `Encapsulated-Node` composite. That is, functions are regarded as a `Composite` and be able to use a variety of APIs (e.g.: `call`, `call->`, `defmodel-as` ...).

In principle, a function takes one argument and returns one value, but by adding more `arguments` the macro automatically wraps the function to satisfy it. For example, `(asnode #'!add 1.0) is transformed into: #'(lambda (x) (!add x 1.0))`. So the first arguments should receive AbstractTensor.

### Usage: call->

It is not elegant to use `call` more than once when composing multiple models.

```lisp
(call (AnyModel1)
      (call (AnyModel2)
             (call (AnyModel3) X)))
```

Instead, you can use the `call->` function:

```lisp
(call-> X
        (AnyModel1)
        (AnyModel2)
        (AnyModel3))
```

However, due to constrains of `call`, it is not possible to place functions here. `asnode` is exactly for this!

```lisp
(call-> X
        (AnyModel1)
        (asnode #'!softmax)
        (asnode #'!view 0) ;; Slicing the tensor: (!view x 0 t ...)
        (asnode #'!add 1.0) ;; X += 1.0
        (asnode !matmul Y) ;; X <- Matmul(X, Y)
        )
```

### Usage2: defmodel-as

The macro `cl-waffe2/vm.nodes:defmodel-as` is able to define new functions/nodes from existing `Composite`. However, this macro only needs the traced computation nodes information to do this. As the simplest case, compiling the AbstractNode `SinNode` (which is callable as `!sin`) into static function, `matrix-sin`.

```lisp
(defmodel-as (asnode #'!sin) :where (A[~] -> B[~]) :asif :function :named matrix-sin)

(matrix-sin (ax+b `(10 10) 0 1)) ;; <- No compiling overhead. Just works like Numpy
```

On a side note: `Encapsulated-Node` itself doesn't provide for `:where` declaration, but you can it with the keyword `:where`.

## [function] call->

```lisp
(call-> input &rest nodes)
```

Starting from `input`, this macro applies a composed function.

```lisp
(call-> (randn `(3 3))       ;; To the given input:
	(asnode #'!add 1.0)  ;;  |
	(asnode #'!relu)     ;;  | Applies operations in this order.
	(asnode #'!sum))))   ;;  ↓
```

`nodes` could be anything as long as the `call` method can handle, but I except node=`Composite`, `AbstractNode`, and `(asnode function ...)`.

## [macro] defsequence

```lisp
(defsequence (name (&rest args) &optional docstring &rest nodes))
```

Defines a Composite that can be defined only by the `call->` method.

### Inputs

`name[symbol]` defines the new Composite after `name`

`args[list]` a list of arguments that used to initialize nodes. Not for `call`.

`docstring[string]` docstring

### Example

```lisp
(defsequence MLP (in-features)
    "Docstring (optional)"
    (LinearLayer in-features 512)
    (asnode #'!tanh)
    (LinearLayer 512 256)
    (asnode #'!tanh)
    (LinearLayer 256 10))

;; Sequence can receive a single argument.
(call (MLP 786) (randn `(10 786)))
```

Tips: Use `(sequencelist-nth n sequence-model)` to read the nth layer of sequence.

## [function] show-backends

```lisp
(show-backends &key (stream t))
```

collects and displays the current state of devices to the given `stream`

### Example

```lisp
(show-backends)

─────[All Backends Tree]──────────────────────────────────────────────────

[*]CPUTENSOR: OpenBLAS=available *simd-extension-p*=available
    └[-]JITCPUTENSOR: compiler=gcc flags=(-fPIC -O3 -march=native) viz=NIL

[*]LISPTENSOR: Common Lisp implementation on matrix operations
    └[-]JITLISPTENSOR: To be deleted in the future release. do not use this.

[-]SCALARTENSOR: is a special tensor for representing scalar values.
    └[-]JITCPUSCALARTENSOR: Use with JITCPUTensor

([*] : in use, [-] : not in use.)
Add a current-backend-state method to display the status.
─────[*using-backend*]───────────────────────────────────────────────────

Priority: Higher <───────────────────>Lower
                  CPUTENSOR LISPTENSOR 

(use with-devices macro or set-devices-toplevel function to change this parameter.)
```

## [function] set-devices-toplevel

```lisp
(set-devices-toplevel &rest devices)
```

Declares devices to use.

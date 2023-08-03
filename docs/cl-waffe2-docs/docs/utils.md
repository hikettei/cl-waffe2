
# [package] cl-waffe2
The package `:cl-waffe2` provides a wide range of utilities.
## Accessing AbstractTensor as an array of other types.
we provide Common utils to access the storage vector of `AbstractTensor` with multiple devices. In addition, those utils endeavour to synchronize the matrix elements as much as possible before and after the conversation.

## [generic] convert-tensor-facet

```lisp
(convert-tensor-facet from to)
```

The generic function `convert-tensor-facet` pays an important role when converting the data structure between `AbstractTensor` and other arrays (e.g.: `simple-array` etc...). Set `from` = `<<Array Before Converted>>`, and `to` = `(type-of <<Datatype you need>>)`, we dispatch the appropriate method and return converted arrays. Note that there's no assurance but before and after the converting, the pointers endeavour to indicate the same thing. If `AbstractTensor` to be converted has shuffled or viewed, we make a copy so that they become contiguous in memory.

This method is intended to be extended by users.

For example, converting `AbstractTensor` -> `simple-array`:

```lisp
(convert-tensor-facet (randn `(3 3)) 'simple-array)
```

See also: `convert-facet`


## [function] change-facet

```lisp
(change-facet (array-from &key (direction 'array)))
```

Changes the facet of given `array-from` into `direction`. This function is just an alias for `convert-tensor-facet`

See also: `convert-tensor-facet`

### direction

As of this writing(2023/7/18), we provide these directions in default.

`array` returns ommon Lisp Array, with keeping the shape of tensors.

`simple-array` returns Common Lisp Array but 1D. the order of elements hinge on the order of `tensor.`

`AbstractTensor` returns `AbstractTensor` (devices to use depend on `*using-device*`). The dtype of returned tensor can be inferred from a first element of given array.


## [macro] with-facet

```lisp
(with-facet (var (object-from &key (direction 'simple-array)) &body body))
```

The macro `with-facet` changes the facet of given `object-from` into `direction`, binding the result to `var`. If you want to apply modifications to `object-from` which applied inside `body`, set `sync`=`t`. (Only available when `object-from`=`AbstractTensor` otherwise ignored).

The macro `with-facet` is working on the flowchart below. Note that on some conditions, `(convert-tensor-facet)` will create an additional copy/compiling which may cause performance issue.

```lisp
[macro with-facet]
        ↓
[Set var <- (convert-tensor-facet object-from direction)] ⚠️ If tensor is viewed/permuted, an additional compiling is invoked!
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

with-facet but input-forms are several.


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

## Brief network description of the configurations
(TODO)
## Sequential Model
(TODO) Composing several layers...
## Trainer
(TODO)

```lisp
minimize!:
  ...


set-input:
  describe ...

predict:
  describe ..
```
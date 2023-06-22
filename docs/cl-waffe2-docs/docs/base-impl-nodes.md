
# Standard Nodes

## [node] ADDNODE

```
(A[~] B[~] -> A[~])
```

### Description

`AddNode` is a node which computes following operation element-wise.

Let X and Y be a given arguments and both are matrix.

```math
X\gets{X + Y}
```

### Constructor

```
(AddNode dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)



### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (values (!move dx dout) (!move dy dout)))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SUBNODE

```
(A[~] B[~] -> A[~])
```

### Description

`SubNode` is a node which computes following operation element-wise.

Let X and Y be a given arguments and both are matrix.

```math
X\gets{X - Y}
```

### Constructor

```
(SubNode dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)



### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (values (!move dx dout) (!move dy (!mul -1 dout))))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] MULNODE

```
(A[~] B[~] -> A[~])
```

### Description

`MulNode` is a node which computes following operation element-wise.

Let X and Y be a given arguments and both are matrix.

```math
X\gets{X * Y}
```

### Constructor

```
(MulNode dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)



### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (values (!mul dout dy) (!mul dout dx)))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] DIVNODE

```
(A[~] B[~] -> A[~])
```

### Description

`DivNode` is a node which computes following operation element-wise.

Let X and Y be a given arguments and both are matrix.

```math
X\gets{X / Y}
```

### Constructor

```
(DivNode dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)



### Backward

✅ Already defined. 

```lisp
((self dout dx dy)
 (values (!div dout dy) (!div (!mul dx (!mul -1 dout)) (!square dy))))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] INVERSETENSORNODE

```
(A[~] -> A[~])
```

### Description

InverseTensorNode is a node which computes following operation element-wise

```math
A\gets{1 / A}
```

### Constructor

```
(InverseTensorNode dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)



### Backward

✅ Already defined. 

```lisp
((self dout dx) (values (!div (!mul -1 dout) (!square dx))))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALARADD

```
(A[SCAL] B[~] -> B[~] WHERE SCAL = 1)
```

### Description

ScalarAdd is a node which computes following operation element-wise.

Let X be a given matrix and S be a given scalar.

```math
X\gets{X + scalar}
```

### Constructor

```
(ScalarAdd dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)


### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (declare (ignore dx dy)) (values (->scal (!mean dout)) dout))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALARSUB

```
(A[SCAL] B[~] -> B[~] WHERE SCAL = 1)
```

### Description

ScalarSub is a node which computes following operation element-wise.

Let X be a given matrix and S be a given scalar.

```math
X\gets{X - scalar}
```

### Constructor

```
(ScalarSub dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)


### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (declare (ignore dx dy))
 (values (->scal (!mul -1.0 (!mean dout))) dout))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALARMUL

```
(A[SCAL] B[~] -> B[~] WHERE SCAL = 1)
```

### Description

ScalarMul is a node which computes following operation element-wise.

Let X be a given matrix and S be a given scalar.

```math
X\gets{X * scalar}
```

### Constructor

```
(ScalarMul dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)


### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (values (->scal (!mean (!mul dy dout))) (!mul dout dx)))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALARDIV

```
(A[SCAL] B[~] -> B[~] WHERE SCAL = 1)
```

### Description

ScalarDiv is a node which computes following operation element-wise.

Let X be a given matrix and S be a given scalar.

```math
X\gets{X / scalar}
```

### Constructor

```
(ScalarDiv dtype)
```

`dtype` dtype to use, being used to dispatch backends. (e.g.: `:float` `:uint8`)


### Backward

✅ Already defined. 

```lisp
((self dout dx dy)
 (values (->scal (!mean (!div (!mul dy (!mul -1 dout)) (!square dx))))
         (!div dout dx)))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] MOVETENSORNODE

```
(A[~] B[~] -> A[~])
```

### Description


Move all the visible elements of `B` into visible areas of `A`.

```math
A\gets{B}
```

### Constraints

In order to implement the behaviour for compilers of eliminating unused copies, all the implementations must satisfy as follows:

On forward:

1. If (movetensor-ignore-me self) is t, return `B` without doing anything.

2. Otherwise, Move all the visible elements of `B` into `A`, and return `A`.

(Note that: until `(tensor-vec A)` is called, `A` is never allocated.)

### Constructor

`(MoveTensorNode dtype)`

`dtype` dtype to use.



### Backward

✅ Already defined. 

```lisp
((self dout dx dy)
 (let ((dy-out
        (if (and (eql (tensor-attribute dy) chain) (movetensor-ignore-me self))
            dout
            (!copy dout))))
   (values
    (if (eql (tensor-attribute dx) chain)
        (!move dx dout)
        dout)
    (if (eql (tensor-attribute dy) chain)
        (!move dy dy-out)
        dy-out))))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
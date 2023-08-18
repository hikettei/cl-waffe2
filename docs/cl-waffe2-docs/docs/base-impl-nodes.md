
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
((self dout dx dy) (declare (ignore dx dy)) (values dout dout))
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
((self dout dx dy) (declare (ignore dx dy)) (values dout (!mul -1 dout)))
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
(A[~] SCALAR[SCAL] -> A[~] WHERE SCAL = 1)
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
((self dout dx dy) (declare (ignore dx dy)) (values dout (->scal (!mean dout))))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALARSUB

```
(A[~] SCALAR[SCAL] -> A[~] WHERE SCAL = 1)
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
 (values dout (->scal (!mul -1.0 (!mean dout)))))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALARMUL

```
(A[~] SCALAR[SCAL] -> A[~] WHERE SCAL = 1)
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
((self dout dx dy) (values (!mul dout dy) (->scal (!mean (!mul dx dout)))))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALARDIV

```
(A[~] SCALAR[SCAL] -> A[~] WHERE SCAL = 1)
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
 (values (!div dout dy)
         (->scal (!mean (!div (!mul dx (!mul -1 dout)) (!square dy))))))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] MOVETENSORNODE

```
(A[~] B[~] -> A[~])
```

### Description


Moves all the visible elements of `B` into visible areas of `A`.

```math
A\gets{B}
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


### Backward

✅ Already defined. 

```lisp
((self dout dx dy)
 (let ((dy-out
        (if (and (eql (tensor-attribute dy) chain) (movetensor-ignore-me self))
            dout
            (if (tensor-permuted-p dout)
                (let ((out
                       (make-input (shape dx) nil create-from dout dtype
                                   (dtype dx) order (order dx))))
                  (!move out dout force t))
                (!copy dout force t)))))
   (values nil dy-out)))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] ABSNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `ABSNODE` takes X as an argument, applying a abs function into each element and writes the result into out.

```math
OUT\gets{abs(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-ABSNODE` `!abs`

### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (declare (ignore dy)) (values (!mul dout (!sign dx)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-ABSNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-ABSNODE takes scalar X as an argument, applying a abs function into each element and writes the result into out.

```math
out\gets{abs(x)}
```
save-for-backward: (T NIL)

See also: `ABSNODE` `!abs`

### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (declare (ignore dy)) (values (!mul dout (!sign dx)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SIGNNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `SIGNNODE` takes X as an argument, applying a sign function into each element and writes the result into out.

```math
OUT\gets{sign(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-SIGNNODE` `!sign`

### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (declare (ignore dout dy)) (values (!mul dx 0) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-SIGNNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-SIGNNODE takes scalar X as an argument, applying a sign function into each element and writes the result into out.

```math
out\gets{sign(x)}
```
save-for-backward: (T NIL)

See also: `SIGNNODE` `!sign`

### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (declare (ignore dout dy)) (values (!mul dx 0) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SQRTNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `SQRTNODE` takes X as an argument, applying a sqrt function into each element and writes the result into out.

```math
OUT\gets{sqrt(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-SQRTNODE` `!sqrt`

### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (declare (ignore dy)) (values (!mul dout (!div 1 dx)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-SQRTNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-SQRTNODE takes scalar X as an argument, applying a sqrt function into each element and writes the result into out.

```math
out\gets{sqrt(x)}
```
save-for-backward: (T NIL)

See also: `SQRTNODE` `!sqrt`

### Backward

✅ Already defined. 

```lisp
((self dout dx dy) (declare (ignore dy)) (values (!mul dout (!div 1 dx)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SQUARENODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `SQUARENODE` takes X as an argument, applying a square function into each element and writes the result into out.

```math
OUT\gets{square(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-SQUARENODE` `!square`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out)) (values (!mul dout x) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-SQUARENODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-SQUARENODE takes scalar X as an argument, applying a square function into each element and writes the result into out.

```math
out\gets{square(x)}
```
save-for-backward: (T NIL)

See also: `SQUARENODE` `!square`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out)) (values (!mul dout x) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SINNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `SINNODE` takes X as an argument, applying a sin function into each element and writes the result into out.

```math
OUT\gets{sin(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-SINNODE` `!sin`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out)) (values (!mul dout (!cos x)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-SINNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-SINNODE takes scalar X as an argument, applying a sin function into each element and writes the result into out.

```math
out\gets{sin(x)}
```
save-for-backward: (T NIL)

See also: `SINNODE` `!sin`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out)) (values (!mul dout (!cos x)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] COSNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `COSNODE` takes X as an argument, applying a cos function into each element and writes the result into out.

```math
OUT\gets{cos(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-COSNODE` `!cos`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!mul -1 (!sin x))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-COSNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-COSNODE takes scalar X as an argument, applying a cos function into each element and writes the result into out.

```math
out\gets{cos(x)}
```
save-for-backward: (T NIL)

See also: `COSNODE` `!cos`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!mul -1 (!sin x))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] TANNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `TANNODE` takes X as an argument, applying a tan function into each element and writes the result into out.

```math
OUT\gets{tan(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-TANNODE` `!tan`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!mul (!cos x) (!cos x)))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-TANNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-TANNODE takes scalar X as an argument, applying a tan function into each element and writes the result into out.

```math
out\gets{tan(x)}
```
save-for-backward: (T NIL)

See also: `TANNODE` `!tan`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!mul (!cos x) (!cos x)))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] ASINNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `ASINNODE` takes X as an argument, applying a asin function into each element and writes the result into out.

```math
OUT\gets{asin(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-ASINNODE` `!asin`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!sqrt (!sub 1 (!square x))))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-ASINNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-ASINNODE takes scalar X as an argument, applying a asin function into each element and writes the result into out.

```math
out\gets{asin(x)}
```
save-for-backward: (T NIL)

See also: `ASINNODE` `!asin`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!sqrt (!sub 1 (!square x))))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] ACOSNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `ACOSNODE` takes X as an argument, applying a acos function into each element and writes the result into out.

```math
OUT\gets{acos(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-ACOSNODE` `!acos`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div -1 (!sqrt (!sub 1 (!square x))))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-ACOSNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-ACOSNODE takes scalar X as an argument, applying a acos function into each element and writes the result into out.

```math
out\gets{acos(x)}
```
save-for-backward: (T NIL)

See also: `ACOSNODE` `!acos`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div -1 (!sqrt (!sub 1 (!square x))))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] ATANNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `ATANNODE` takes X as an argument, applying a atan function into each element and writes the result into out.

```math
OUT\gets{atan(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-ATANNODE` `!atan`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!add 1 (!square x)))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-ATANNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-ATANNODE takes scalar X as an argument, applying a atan function into each element and writes the result into out.

```math
out\gets{atan(x)}
```
save-for-backward: (T NIL)

See also: `ATANNODE` `!atan`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!add 1 (!square x)))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SINHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `SINHNODE` takes X as an argument, applying a sinh function into each element and writes the result into out.

```math
OUT\gets{sinh(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-SINHNODE` `!sinh`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out)) (values (!mul dout (!cosh x)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-SINHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-SINHNODE takes scalar X as an argument, applying a sinh function into each element and writes the result into out.

```math
out\gets{sinh(x)}
```
save-for-backward: (T NIL)

See also: `SINHNODE` `!sinh`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out)) (values (!mul dout (!cosh x)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] COSHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `COSHNODE` takes X as an argument, applying a cosh function into each element and writes the result into out.

```math
OUT\gets{cosh(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-COSHNODE` `!cosh`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!mul -1 (!sinh x))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-COSHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-COSHNODE takes scalar X as an argument, applying a cosh function into each element and writes the result into out.

```math
out\gets{cosh(x)}
```
save-for-backward: (T NIL)

See also: `COSHNODE` `!cosh`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!mul -1 (!sinh x))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] TANHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `TANHNODE` takes X as an argument, applying a tanh function into each element and writes the result into out.

```math
OUT\gets{tanh(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-TANHNODE` `!tanh`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!mul (!cosh x) (!cosh x)))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-TANHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-TANHNODE takes scalar X as an argument, applying a tanh function into each element and writes the result into out.

```math
out\gets{tanh(x)}
```
save-for-backward: (T NIL)

See also: `TANHNODE` `!tanh`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!mul (!cosh x) (!cosh x)))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] ASINHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `ASINHNODE` takes X as an argument, applying a asinh function into each element and writes the result into out.

```math
OUT\gets{asinh(X)}
```

save-for-backward: NIL

See also: `SCALAR-ASINHNODE` `!asinh`

### Backward

❌ Undefined. (To make it differentiable, must be defined with `define-impl` macro.)
## [node] SCALAR-ASINHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-ASINHNODE takes scalar X as an argument, applying a asinh function into each element and writes the result into out.

```math
out\gets{asinh(x)}
```
save-for-backward: NIL

See also: `ASINHNODE` `!asinh`

### Backward

❌ Undefined. (To make it differentiable, must be defined with `define-impl` macro.)
## [node] ACOSHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `ACOSHNODE` takes X as an argument, applying a acosh function into each element and writes the result into out.

```math
OUT\gets{acosh(X)}
```

save-for-backward: NIL

See also: `SCALAR-ACOSHNODE` `!acosh`

### Backward

❌ Undefined. (To make it differentiable, must be defined with `define-impl` macro.)
## [node] SCALAR-ACOSHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-ACOSHNODE takes scalar X as an argument, applying a acosh function into each element and writes the result into out.

```math
out\gets{acosh(x)}
```
save-for-backward: NIL

See also: `ACOSHNODE` `!acosh`

### Backward

❌ Undefined. (To make it differentiable, must be defined with `define-impl` macro.)
## [node] ATANHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `ATANHNODE` takes X as an argument, applying a atanh function into each element and writes the result into out.

```math
OUT\gets{atanh(X)}
```

save-for-backward: NIL

See also: `SCALAR-ATANHNODE` `!atanh`

### Backward

❌ Undefined. (To make it differentiable, must be defined with `define-impl` macro.)
## [node] SCALAR-ATANHNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-ATANHNODE takes scalar X as an argument, applying a atanh function into each element and writes the result into out.

```math
out\gets{atanh(x)}
```
save-for-backward: NIL

See also: `ATANHNODE` `!atanh`

### Backward

❌ Undefined. (To make it differentiable, must be defined with `define-impl` macro.)
## [node] EXPNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `EXPNODE` takes X as an argument, applying a exp function into each element and writes the result into out.

```math
OUT\gets{exp(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-EXPNODE` `!exp`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out)) (values (!mul dout (!exp x)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-EXPNODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-EXPNODE takes scalar X as an argument, applying a exp function into each element and writes the result into out.

```math
out\gets{exp(x)}
```
save-for-backward: (T NIL)

See also: `EXPNODE` `!exp`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out)) (values (!mul dout (!exp x)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] LOG2NODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `LOG2NODE` takes X as an argument, applying a log2 function into each element and writes the result into out.

```math
OUT\gets{log2(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-LOG2NODE` `!log2`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!mul x (log 2)))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-LOG2NODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-LOG2NODE takes scalar X as an argument, applying a log2 function into each element and writes the result into out.

```math
out\gets{log2(x)}
```
save-for-backward: (T NIL)

See also: `LOG2NODE` `!log2`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!mul x (log 2)))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] LOG10NODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `LOG10NODE` takes X as an argument, applying a log10 function into each element and writes the result into out.

```math
OUT\gets{log10(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-LOG10NODE` `!log10`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!mul x (log 10)))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-LOG10NODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-LOG10NODE takes scalar X as an argument, applying a log10 function into each element and writes the result into out.

```math
out\gets{log10(x)}
```
save-for-backward: (T NIL)

See also: `LOG10NODE` `!log10`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out))
 (values (!mul dout (!div 1 (!mul x (log 10)))) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] LOGENODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node `LOGENODE` takes X as an argument, applying a loge function into each element and writes the result into out.

```math
OUT\gets{loge(X)}
```

save-for-backward: (T NIL)

See also: `SCALAR-LOGENODE` `!loge`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out)) (values (!mul dout (!div 1 x)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] SCALAR-LOGENODE

```
(X[~] OUT[~] -> OUT[~])
```

### Description

The node SCALAR-LOGENODE takes scalar X as an argument, applying a loge function into each element and writes the result into out.

```math
out\gets{loge(x)}
```
save-for-backward: (T NIL)

See also: `LOGENODE` `!loge`

### Backward

✅ Already defined. 

```lisp
((self dout x out) (declare (ignore out)) (values (!mul dout (!div 1 x)) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] LAZYTRANSPOSENODE

```
(A[~ I J] -> A[~ I J])
```

### Description

LazyTransposeNode is a matmul-dedicated node to implement zero-cost transpose.

The node stores untransposed tensor at `raw-tensor`, when expanding matmul form, you can read it if needed.

### Backward

✅ Already defined. 

```lisp
((self dout dx) (declare (ignore dx)) (values dout))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] ARGMAX-NODE

```
(A[~] OUT[OUT-SIZE] -> OUT[OUT-SIZE])
```

### Description

ArgMax-Node finds an index of maximum value of all elements in A. `OUT` is overwritten with the result.

A is a target to find a maximum value, and OUT is a place to set the index.

### Constructor

```
(ArgMax-Node out-size)
```

`out-size` the reducted shape of `out`.


### Backward

✅ Already defined. 

```lisp
((self dout da do) (declare (ignore dout da do)) (values nil nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] ARGMIN-NODE

```
(A[~] OUT[OUT-SIZE] -> OUT[OUT-SIZE])
```

### Description

ArgMin-Node finds an index of minimum value of all elements in A. `OUT` is overwritten with the result.

A is a target to find a minimum value, and OUT is a place to set the index.

### Constructor

```
(ArgMin-Node out-size)
```

`out-size` the reducted shape of `out`.

### Backward

✅ Already defined. 

```lisp
((self dout da do) (declare (ignore dout da do)) (values nil nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] MAXVALUE-NODE

```
(A[~] OUT[OUT-SIZE] -> OUT[OUT-SIZE])
```

### Description

MaxValue-Node finds a maximum value of all elements in A. `OUT` is overwritten with the result.

A is a target to find a maximum value, and OUT is a place to set the index.

### Constructor

```
(MaxValue-Node out-size)
```

`out-size` the reducted shape of `out`.


### Backward

✅ Already defined. 

```lisp
((self dout da do) (declare (ignore do))
 (let ((mask (a=b da (!view (!max da) (broadcast-to da)))))
   (values (!mul mask (!view dout (broadcast-to mask))) nil)))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] MINVALUE-NODE

```
(A[~] OUT[OUT-SIZE] -> OUT[OUT-SIZE])
```

### Description

MinValue-Node finds a minimum value of all elements in A. `OUT` is overwritten with the result.

A is a target to find a minimum value, and OUT is a place to set the index.

### Constructor

```
(MinValue-Node out-size)
```

`out-size` the reducted shape of `out`.

### Backward

✅ Already defined. 

```lisp
((self dout da do) (declare (ignore do))
 (let ((mask (a=b da (!view (!min da) (broadcast-to da)))))
   (values (!mul mask (!view dout (broadcast-to mask))) nil)))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] MATMULNODE

```
(A[~ I J] B[~ J K] C[~ I K] -> C[~ I K])
```

### Description

MatmulNode Computes a matrix multiplication of given A and B, set the result to C.

```math
C\gets{gemm(1.0, A, B, 0.0, C)}
```

### Constructor

```
(MatMulNode dtype &key transpose-a transpose-b)
```

`dtype` dtype to use.

`transpose-a transpose-b[boolean]` becomes t if the given `a` or `b` needs to be transposed respectively. call `(read-untransposed tensor)` to read untransposed tensor.



### Backward

✅ Already defined. 

```lisp
((self dout da db do) (declare (ignore do))
 (values (!matmul dout (!t db)) (!matmul (!t da) dout) nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] WHERE-OPERATION-NODE

```
(A[~] OUT[~] -> OUT[~])
```

### Description

Where-Operation-Node is a node which set `true-then`, if the result of calling `condition` with each element of A, is t and if it is NIL, set `false-then` at corresponding position.

### Constructor

```
(Where-Operation-Node condition true-then false-then)
```

`true-then` and `false-then` is a number.

`condition` a single argument function, each element of A is argument. (e.g.: this could be `#'evenp` `#'oddp` etc...)


### Backward

✅ Already defined. 

```lisp
((self dout da do) (declare (ignore dout da do)) (values nil nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
## [node] COMPARE-OPERATION-NODE

```
(A[~] B[~] OUT[~] -> OUT[~])
```

### Description

Compare-Operation-Node is a node which set `true-then`, if the result of calling `condition` with each element of A and B, if it is NIl set `false-then` at corresponding position.

### Constructor

```
(Compare-Operation-Node condition true-then false-then)
```

`true-then` and `false-then` is a number.

`condition` a two arguments function, each element of A and B is argument. (e.g.: this could be `#'>` or `#'<` etc...)


### Backward

✅ Already defined. 

```lisp
((self dout da db do) (declare (ignore dout da db do)) (values nil nil nil))
```

No need to implement backwards at `define-impl`. (they'd be ignored.)
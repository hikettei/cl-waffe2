
# Basic APIs

## [function] !matrix-add

```lisp
(!matrix-add x y)
```

The function `!matrix-add` calls `ADDNODE` and adds X and Y element-wise, returning a new tensor.

```math
X_{copy}\gets{X + Y}
```

### Inputs

`X` and `Y` must be a AbstractTensor (not a ScalarTensor), with the same shape.

### SideEffects

None.

## [function] !matrix-sub

```lisp
(!matrix-sub x y)
```

The function `!matrix-sub` calls `SUBNODE` and substracts X by Y element-wise, returning a new tensor.

```math
X_{copy}\gets{X - Y}
```

### Inputs

`X` and `Y` must be a AbstractTensor (not a ScalarTensor), with the same shape.

### SideEffects

None.

## [function] !matrix-mul

```lisp
(!matrix-mul x y)
```

The function `!matrix-mul` calls `MULNODE` and multiplies X and Y element-wise, returning a new tensor.

```math
X_{copy}\gets{X * Y}
```

### Inputs

`X` and `Y` must be a AbstractTensor (not a ScalarTensor), with the same shape.

### SideEffects

None.

## [function] !matrix-div

```lisp
(!matrix-div x y)
```

The function `!matrix-div` calls `DIVNODE` and divides X by Y element-wise, returning a new tensor.

```math
X_{copy}\gets{X / Y}
```

### Inputs

`X` and `Y` must be a AbstractTensor (not a ScalarTensor), with the same shape.

### SideEffects

None.
## [function] !inverse

```lisp
(!inverse tensor)
```

The function `!inverse` calls `InverseTensorNode`, and finds the inverse of the received Tensor/Scalar, returning a new tensor.

```math
X_{copy}\gets{1 / X}
```

### Inputs

tensor[ScalarTensor/AbstractTensor/Number]

## [function] !scalar-add

```lisp
(!scalar-add scalar x)
```

The function !SCALAR-ADD computes following operation with calling `SCALARADD`, returning a new tensor.

```math
X_{copy}\gets{X + scalar}
```

### Inputs

`scalar` could be one of `ScalarTensor` or `number`

`tensor` `AbstractTensor` (should not be a scalar)

## [function] !scalar-sub

```lisp
(!scalar-sub scalar x)
```

The function !SCALAR-SUB computes following operation with calling `SCALARSUB`, returning a new tensor.

```math
X_{copy}\gets{X - scalar}
```

### Inputs

`scalar` could be one of `ScalarTensor` or `number`

`tensor` `AbstractTensor` (should not be a scalar)

## [function] !scalar-mul

```lisp
(!scalar-mul scalar x)
```

The function !SCALAR-MUL computes following operation with calling `SCALARMUL`, returning a new tensor.

```math
X_{copy}\gets{X * scalar}
```

### Inputs

`scalar` could be one of `ScalarTensor` or `number`

`tensor` `AbstractTensor` (should not be a scalar)

## [function] !scalar-div

```lisp
(!scalar-div scalar x)
```

The function !SCALAR-DIV computes following operation with calling `SCALARDIV`, returning a new tensor.

```math
X_{copy}\gets{X / scalar}
```

### Inputs

`scalar` could be one of `ScalarTensor` or `number`

`tensor` `AbstractTensor` (should not be a scalar)

## [function] !sas-add

The function !sas-add provides differentiable scalar-and-scalar operation.

Calling a node `SCALARANDSCALARADD`, the function performs following operation:

```math
x_{copy}\gets{x + y}
```

### Inputs

`x` `y` could be one of: `ScalarTensor` or `number`


## [function] !sas-sub

The function !sas-sub provides differentiable scalar-and-scalar operation.

Calling a node `SCALARANDSCALARSUB`, the function performs following operation:

```math
x_{copy}\gets{x - y}
```

### Inputs

`x` `y` could be one of: `ScalarTensor` or `number`


## [function] !sas-mul

The function !sas-mul provides differentiable scalar-and-scalar operation.

Calling a node `SCALARANDSCALARMUL`, the function performs following operation:

```math
x_{copy}\gets{x * y}
```

### Inputs

`x` `y` could be one of: `ScalarTensor` or `number`


## [function] !sas-div

The function !sas-div provides differentiable scalar-and-scalar operation.

Calling a node `SCALARANDSCALARDIV`, the function performs following operation:

```math
x_{copy}\gets{x / y}
```

### Inputs

`x` `y` could be one of: `ScalarTensor` or `number`


## [function] !add

```lisp
(!add x y)
```

The function provides general-purpose arithmetic operation.

Given type of tensors, this function dispatches these functions automatically:

1. `!sas-add`

2. `!scalar-add`

3. `!matrix-add`

### Inputs

`x` `y` could be one of `AbstractTensor` `number` `ScalarTensor`

### SideEffects

None

## [function] !sub

```lisp
(!sub x y)
```

The function provides general-purpose arithmetic operation.

Given type of tensors, this function dispatches these functions automatically:

1. `!sas-sub`

2. `!scalar-sub`

3. `!matrix-sub`

### Inputs

`x` `y` could be one of `AbstractTensor` `number` `ScalarTensor`

### SideEffects

None

## [function] !mul

```lisp
(!mul x y)
```

The function provides general-purpose arithmetic operation.

Given type of tensors, this function dispatches these functions automatically:

1. `!sas-mul`

2. `!scalar-mul`

3. `!matrix-mul`

### Inputs

`x` `y` could be one of `AbstractTensor` `number` `ScalarTensor`

### SideEffects

None

## [function] !div

```lisp
(!div x y)
```

The function provides general-purpose arithmetic operation.

Given type of tensors, this function dispatches these functions automatically:

1. `!sas-div`

2. `!scalar-div`

3. `!matrix-div`

### Inputs

`x` `y` could be one of `AbstractTensor` `number` `ScalarTensor`

### SideEffects

None

## [function] !+

Is the equivalent to just doing `(reduce #'!ADD numbers)`

### Example

```
(#'!ADD 1 2 3 4 5)
```
## [function] !-

Is the equivalent to just doing `(reduce #'!SUB numbers)`

### Example

```
(#'!SUB 1 2 3 4 5)
```
## [function] !*

Is the equivalent to just doing `(reduce #'!MUL numbers)`

### Example

```
(#'!MUL 1 2 3 4 5)
```
## [function] !/

Is the equivalent to just doing `(reduce #'!DIV numbers)`

### Example

```
(#'!DIV 1 2 3 4 5)
```
## [function] !move

```lisp
(!move place tensor)
```

```math
A\gets{B}
```

The function !move returns a node which moves tensor's visible elements into place's visible elements.

### nodes

one of: `MoveTensorNode` `ScalarTensorNode`

### Inputs

`place[AbstractTensor]` tensor to be overwritten.

`tensor[AbstractTensor]` tensor to be referred.

`force[boolean]` If t, the pruning of operation by cl-waffe2 will never done.

### Output

Unevaluated Copied Tensor.
## [function] !copy

```lisp
(!copy tensor)
```

The function !copy returns a node which makes a copy the tensor's visible area.

Note that: the function `!copy` never creates a new tensor larger than (tensor-vec tensor) has, (i.e.: copying broadcasted tensor will return broadcasted and copied tensor).

`!copy` is used to make a cache before calling destructive operation to avoid side effects, therefore if the copy is included to be useless by compiler, this operations is being ignored without changing its behaviour. And this is why !copy returns `InputTensor`, not `AbstractTensor`.

See also: `!copy-force` never being ignored by compiler, and broadcasted axes will be padded.

Input:  Tensor[AbstractTensor]
Output: Tensor[AbstractTensor]
## [function] !copy-force

```lisp
(!copy-force (tensor))
```

The function !copy-force returns a node which copies the given tensor forcibly while the function !copy sometimes ignored.

This function is also used to adjust memory alignment of tensor.
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

## [function] !flatten

```
(!flatten tensor)
```

equivalent to the `(!reshape tensor t)`

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

## [function] ->scal

```
(->scal matrix-tensor)
```

The function ->scal receives `matrix-tensor` with total-size = 1, returning a ScalarTensor.

## [function] ->mat

```
(->mat scalar-tensor &key (dims 1))
```

The function ->mat receives `ScalarTensor`, returning a matrix with the number of axis=dims.
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

## [function] proceed-time

```
(proceed-time tensor)
```

An alias for (proceed tensor :measure-time t)

Note that: the proceed-time function invokes forward function twice times, in order for processing system to trace compiled lisp code, and ignoring allocation time.
## [function] proceed-backward

```
(proceed-backward tensor)
```

The function proceed-backward calls forward and backwrd of the tensor.

### Output

`T` (which indicates backward is succeed)

## [macro] %transform

```lisp
(%transform &body transform-syntax)
```

`%transform` is a macro to describe `!view`, `!permute` and `broadcasting` of the given tensors together in a concise manner. In short word, `%transform = !view + !permute + Broadcasting`. The transformation of tensor are described on the same syntax of `Subscript DSL` but before and after `->`, there is always one tensor for each.

```
(Example)
(%transform A[i j] -> A[j i])
```

The variable names (e.g.: `A`) are exactly the name of the variable used by the `%transform` macro, which must be bound in scope. It is optional to give the name to the tensor after `->`.

```lisp
(defun transpose-revisit (tensor)
    (%transform tensor[~ i j] -> [j i]))
```

### Syntax

Following the rules below, `%transform` calls appropriate functions. If `~` were used after `->`, the macro is expanded into `!flexible ...`, or call `!permute` as long as all symbols appeared before `->` were also used after `->`. Otherwise, call `!view`.

### Adding an broadcastable axis.

The `broadcastable axis` is the range in which `1` of the shape of tensors can be added if needed, and at most one exists in one matrix.

If the subscripts of the tensor after `->` includes `~`, the corresponding position of the shape becomes `broadcastable`.

For example:

```lisp
(%transform A[i j] -> A[~ i j])
(%transform A[~ i j] -> A[~ i j])
```

### Adjustable dimensions

the `~` symbol used before `->` means: the number of dimensions of the corresponding part could be anything.

```lisp
(%transform A[~ i j] -> A[i j]
```

### Shuffling the permution of tensor

If symbols used before `->` are also appeared in after `->`, the corresponding symbols indicate the permution of tensor.

```lisp
(%transform A[i j] -> [j i])
(%transform A[~ i j] -> [j i])
(%transform A[i ~ j] -> [j i]) ;; the same as (!permute a 1 :~ 0)
```

### Make a view of tensors.

Set symbols (which aren't used before `->`) or fixnum to make a index. `(start end)` also creates a slice. Setting characters like `*10` `*a` broadcasts the axis.

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

## [function] !abs

```lisp
(!abs x &key (-> nil))
```

The function !abs takes `x` as an argument, applying a abs function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{abs(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-ABSNODE` `ABSNODE`

### SideEffects

`->` is destructed.

## [function] !sign

```lisp
(!sign x &key (-> nil))
```

The function !sign takes `x` as an argument, applying a sign function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{sign(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-SIGNNODE` `SIGNNODE`

### SideEffects

`->` is destructed.

## [function] !sqrt

```lisp
(!sqrt x &key (-> nil))
```

The function !sqrt takes `x` as an argument, applying a sqrt function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{sqrt(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-SQRTNODE` `SQRTNODE`

### SideEffects

`->` is destructed.

## [function] !square

```lisp
(!square x &key (-> nil))
```

The function !square takes `x` as an argument, applying a square function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{square(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-SQUARENODE` `SQUARENODE`

### SideEffects

`->` is destructed.

## [function] !sin

```lisp
(!sin x &key (-> nil))
```

The function !sin takes `x` as an argument, applying a sin function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{sin(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-SINNODE` `SINNODE`

### SideEffects

`->` is destructed.

## [function] !cos

```lisp
(!cos x &key (-> nil))
```

The function !cos takes `x` as an argument, applying a cos function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{cos(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-COSNODE` `COSNODE`

### SideEffects

`->` is destructed.

## [function] !tan

```lisp
(!tan x &key (-> nil))
```

The function !tan takes `x` as an argument, applying a tan function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{tan(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-TANNODE` `TANNODE`

### SideEffects

`->` is destructed.

## [function] !asin

```lisp
(!asin x &key (-> nil))
```

The function !asin takes `x` as an argument, applying a asin function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{asin(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-ASINNODE` `ASINNODE`

### SideEffects

`->` is destructed.

## [function] !acos

```lisp
(!acos x &key (-> nil))
```

The function !acos takes `x` as an argument, applying a acos function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{acos(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-ACOSNODE` `ACOSNODE`

### SideEffects

`->` is destructed.

## [function] !atan

```lisp
(!atan x &key (-> nil))
```

The function !atan takes `x` as an argument, applying a atan function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{atan(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-ATANNODE` `ATANNODE`

### SideEffects

`->` is destructed.

## [function] !sinh

```lisp
(!sinh x &key (-> nil))
```

The function !sinh takes `x` as an argument, applying a sinh function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{sinh(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-SINHNODE` `SINHNODE`

### SideEffects

`->` is destructed.

## [function] !cosh

```lisp
(!cosh x &key (-> nil))
```

The function !cosh takes `x` as an argument, applying a cosh function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{cosh(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-COSHNODE` `COSHNODE`

### SideEffects

`->` is destructed.

## [function] !tanh

```lisp
(!tanh x &key (-> nil))
```

The function !tanh takes `x` as an argument, applying a tanh function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{tanh(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-TANHNODE` `TANHNODE`

### SideEffects

`->` is destructed.

## [function] !asinh

```lisp
(!asinh x &key (-> nil))
```

The function !asinh takes `x` as an argument, applying a asinh function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{asinh(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-ASINHNODE` `ASINHNODE`

### SideEffects

`->` is destructed.

## [function] !acosh

```lisp
(!acosh x &key (-> nil))
```

The function !acosh takes `x` as an argument, applying a acosh function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{acosh(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-ACOSHNODE` `ACOSHNODE`

### SideEffects

`->` is destructed.

## [function] !atanh

```lisp
(!atanh x &key (-> nil))
```

The function !atanh takes `x` as an argument, applying a atanh function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{atanh(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-ATANHNODE` `ATANHNODE`

### SideEffects

`->` is destructed.

## [function] !exp

```lisp
(!exp x &key (-> nil))
```

The function !exp takes `x` as an argument, applying a exp function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{exp(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-EXPNODE` `EXPNODE`

### SideEffects

`->` is destructed.

## [function] !log2

```lisp
(!log2 x &key (-> nil))
```

The function !log2 takes `x` as an argument, applying a log2 function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{log2(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-LOG2NODE` `LOG2NODE`

### SideEffects

`->` is destructed.

## [function] !log10

```lisp
(!log10 x &key (-> nil))
```

The function !log10 takes `x` as an argument, applying a log10 function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{log10(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-LOG10NODE` `LOG10NODE`

### SideEffects

`->` is destructed.

## [function] !loge

```lisp
(!loge x &key (-> nil))
```

The function !loge takes `x` as an argument, applying a loge function into each element and writes the result into `->`.

```math
OUT_{copy}\gets{loge(X)}
```

(where `OUT` = `->`)

### Inputs

`x` [AbstractTensor or ScalarTensor or number]

`->` (nil or AbstractTensor). the place to set the result. If nil, a new tensor is allocated.

### Returns

`->`

### Nodes

`SCALAR-LOGENODE` `LOGENODE`

### SideEffects

`->` is destructed.

## [function] !sum

```
(!sum tensor &key (axis t) (-> nil) (keepdims nil))
```

The function !sum return a node which computes the sum of tensor along the given axis.

### Inputs

`tensor`, a tensor to be reducted.

`axis`[t or fixnum or list] the axis to be reducted. (-1, -2... is ok)

`->` [AbstractTensor or nil] the place to set the result. If nil, creates a new tensor.

`dims`[boolean] If t, the axis reducted is broadcasted.

Return:

`->`[AbstractTensor] the result.
## [function] !mean

```
(!mean tensor &key (axis t) (-> nil) (keepdims nil))
```

The function !mean return a node which computes the average of tensor along the given axis.

### Inputs

`tensor`, a tensor to be reducted.

`axis`[t or fixnum or list] the axis to be reducted. (-1, -2... is ok)

`->` [AbstractTensor or nil] the place to set the result. If nil, creates a new tensor.

`keepdims` [boolean] If t, the axis reducted is broadcasted.

### Return

`->`[AbstractTensor] the result.

## [function] !argmax

```
(!argmax tensor &key (axis -1) (out nil))
```

The function !argmax computes the indices of maximum values of all elements below the **axis** dimension in the given tensor.

### Inputs

`tensor`

`axis`

`out`

### Returns

AbstractTensor[uint32] with dimensions behind `axis` is replaced with 1.



## [function] !argmin

```
(!argmin tensor &key (axis -1) (out nil))
```

The function !argmin computes the indices of minimum values of all elements below the **axis** dimension in the given tensor.

### Inputs

`tensor`

`axis`

`out`

### Returns

AbstractTensor[uint32] with dimensions behind `axis` is replaced with 1.

## [function] !max

```
(!max tensor &key (axis -1) (out nil))
```

The function `!max` finds largest values of all elements below the **axis** rank in the given tensor.

### Inputs

`tensor`

`axis`

`out`

### Returns

`AbstractTensor` with dimensions behind `axis` is replaced with 1.

## [function] !min

```
(!min tensor &key (axis -1) (out nil))
```

The function `!min` finds the smallest values of all elements below the **axis** rank in the given tensor.

### Inputs

`tensor`

`axis`

`out`

### Returns

`AbstractTensor` with dimensions behind `axis` is replaced with 1.

## [function] !t

```
(!t tensor)
```

Transposes the last two axes of the given tensor.

When called with !matmul, the operation is ignored.

## [function] !matmul

```lisp
(!matmul x y &key (out nil) (transpose-x nil) (transpose-y nil))
```

Computing a matrix multiplication of X and Y. The result is stored in out if specified, otherwise creates a new tensor.

```math
out\gets{gemm(1.0, x, y, 0.0, out)}
```

### Inputs

`transpose-x, transpose-y[boolean]` If t, the inputs are wrapped with `(!t tensor)`.

### Tips: Lazy-Transpose-Node

If the last backward of given arguments are `LazyTransposeNode` (created with the function `!t`), the function `!matmul` will transpose them without making a copy (i.e.: zero-cost transpose). In any other case (the last two dimensions' permution, or view are too complicated), `!matmul` will produce an additional copy for fast computing.


## [function] !dot

```
(!dot x y)
```

Finds a dot product of x and y. Unlike `numpy.dot`, `!dot` intentionally only supports computing the dot product of two 1D tensors with the same number of elements.

```lisp
(proceed (!dot (randn `(100)) (randn `(10 10))))
{CPUTENSOR[float] :shape (1) -> :view (<0>) -> :visible-shape (1) :named ChainTMP115880 
  :vec-state [computed]
  (21.594929)
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```
## [function] !where

```
(!where tensor condition &key (true-then 1) (false-then 0) (out nil))
```

The function !where returns a elements selected-from `true-then` or `false-then`, depending on condition.

The operation is defined as:

```math
\begin{equation}
  out_i=
  \begin{cases}
    \text{true-then} & condition(X_i) \\
    \text{false-then} & \text{otherwise}
  \end{cases}
\end{equation}
```

(where X = tensor)

### Inputs

`out` place to set the result
`condition` an funcallable function. (e.g.: #'evenp #'oddp etc...)

## [function] !where

```
(!compare tensor1 tensor2 condition &key (true-then 1) (false-then 0) (out nil))
```

The function !compare returns a elements selected-from `true-then` or `false-then`, depending on condition.

The operation is defined as:

```math
\begin{equation}
  out_i=
  \begin{cases}
    \text{true-then} & condition(X_i, Y_i) \\
    \text{false-then} & \text{otherwise}
  \end{cases}
\end{equation}
```

(where X = tensor1, Y=tensor2)

### Inputs

`out` place to set the result
`condition` an funcallable function. (e.g.: #'> #'< etc...)
## [function] a>scal

```
(a>scal A scal &key (out nil) (true-then 1) (false-then 0))
```

The function a>scal sets `true-then` if the equation: `element > scal` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` AbstractTensor
`scal` number (not a ScalarTensor)

(TODO: ScalarTensor as scal)
## [function] a<scal

```
(a<scal A scal &key (out nil) (true-then 1) (false-then 0))
```

The function a<scal sets `true-then` if the equation: `element < scal` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` AbstractTensor
`scal` number (not a ScalarTensor)

(TODO: ScalarTensor as scal)
## [function] a>=scal

```
(a>=scal A scal &key (out nil) (true-then 1) (false-then 0))
```

The function a>=scal sets `true-then` if the equation: `element >= scal` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` AbstractTensor
`scal` number (not a ScalarTensor)

(TODO: ScalarTensor as scal)
## [function] a<=scal

```
(a<=scal A scal &key (out nil) (true-then 1) (false-then 0))
```

The function a<=scal sets `true-then` if the equation: `element <= scal` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` AbstractTensor
`scal` number (not a ScalarTensor)

(TODO: ScalarTensor as scal)
## [function] a=scal

```
(a=scal A scal &key (out nil) (true-then 1) (false-then 0))
```

The function a=scal sets `true-then` if the equation: `element = scal` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` AbstractTensor
`scal` number (not a ScalarTensor)

(TODO: ScalarTensor as scal)
## [function] a>b

```
(a>b A B &key (out nil) (true-then 1) (false-then 0))
```

The function a>b sets `true-then` if the equation: `A > B` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` `B` AbstractTensor to be compared.

## [function] a<b

```
(a<b A B &key (out nil) (true-then 1) (false-then 0))
```

The function a<b sets `true-then` if the equation: `A < B` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` `B` AbstractTensor to be compared.

## [function] a>=b

```
(a>=b A B &key (out nil) (true-then 1) (false-then 0))
```

The function a>=b sets `true-then` if the equation: `A >= B` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` `B` AbstractTensor to be compared.

## [function] a<=b

```
(a<=b A B &key (out nil) (true-then 1) (false-then 0))
```

The function a<=b sets `true-then` if the equation: `A <= B` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` `B` AbstractTensor to be compared.

## [function] a=b

```
(a=b A B &key (out nil) (true-then 1) (false-then 0))
```

The function a=b sets `true-then` if the equation: `A = B` is t, otherwise set `false-then` at the corresponding positions.

### Inputs

`A` `B` AbstractTensor to be compared.

## [function] padding

```lisp
(padding tensor pad-width &key (pad-maker #'ax+b) (initargs `(0 0)))
```

Creating a new InputTensor with shape after padding, the function `padding` moves the given tensor into a new area.

### Implementation

```lisp

(padding (ax+b `(1 3) 0 1) `((1 1) (1 1)))

[Corresponds with...]

                    00000
+++ -> [padding] -> 0+++0
                    00000
```

The operation is performed in the following steps:

First, creates a new tensor with the shape of after padded which is initialized via the `pad-maker` function, where `pad-maker` is an initializer function, that is, functions defined by `define-initializer-function` or exported from the `:cl-waffe2/distribution` package.


```lisp
+++++
+++++
+++++
```

The function `padding` uses the form below to initialize tensors.

```lisp
(apply pad-maker initargs)
```

In default, `(apply #'ax+b `(0 0))`.

Second, makes a view of the new tensor and match its shape to the base tensor. the argument `pad-width` is used to determine offsets of each axis. `pad-width` is the number of values to the edges of each axis. and given as: `((before_1 after_1) (before_2 after_2) ...)`. `0~before_n` and `before_n~last` are the subject to be padded.

```lisp
+++++              -----
+++++ -> [view] -> -+++-
+++++              -----
                       ^ after_2
+ ... visible area
- ... hide area by view
```

Finally, moves all elements in the base tensor into viewed tensor, and later reset the view.

### Inputs

`tensor[AbstractTensor]` tensor to be padded.

`pad-width[list]` the number of the edges and given as: `((before_1 after_1) (before_2 after_2) ...)`. the forward is the same as [np.pad](https://numpy.org/doc/stable/reference/generated/numpy.pad.html). Set t instead of `(before_n after_n)` and ignores the corresponding position of axis.
 
`pad-maker[function]` an initializer-function

`initargs[list]` a list of arguments for `pad-maker`

Note that: the axes to be padded, must be fixnum. not a symbol.

If the shapes does not change before/after padding, returns the given tensor as it is.

### Example

```lisp
(proceed (padding (ax+b `(3 3) 0 1) `((1 1) (1 1))))

{CPUTENSOR[float] :shape (5 5) -> :view (<T> <T>) -> :visible-shape (5 5) :named ChainTMP1104579 
  :vec-state [computed]
  ((0.0 0.0 0.0 0.0 0.0)           
   (0.0 1.0 1.0 1.0 0.0)   
        ...
   (0.0 1.0 1.0 1.0 0.0)
   (0.0 0.0 0.0 0.0 0.0))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] broadcast-to

Returns the subscript of the `!view` that is broadcasting to be the same shape as the `object-tensor`.

For example:

```lisp
;; x                ... ( 3 3 ) Tensor
;; (!sum x :axis 1) ... ( 3 1 ) Tensor

;; broadcast-to will return: (t `(:broadcast 3))
(!mul x (!view (!sum x :axis 1) (broadcast-to x)))
```

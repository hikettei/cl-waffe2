
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


# cl-waffe2/nn

## [function] !relu

Returns a tensor that applied ReLU element-wise.

```math
ReLU(x) = max(x, 0)
```
### Example

```lisp
(proceed (!relu (randn `(10 10))))

{CPUTENSOR[float] :shape (10 10) :named ChainTMP13108 
  :vec-state [computed]
  ((-0.0        -0.0        -0.0        ~ 0.2979711   -0.0        0.77263427)                   
   (1.6693149   -0.0        -0.0        ~ 0.22997802  1.289548    1.6337485)   
                ...
   (0.10559099  0.1584125   -0.0        ~ 1.0346866   0.32413962  -0.0)
   (1.9992205   1.6432109   -0.0        ~ 0.8558499   0.7596627   -0.0))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] !gelu

Not Implemented Yet.

## [function] !sigmoid

Returns a tensor that applied sigmoid function element-wise.

```math
Sigmoid(x) = \frac{1}{1 + exp(-x)}
```

### Example

```lisp
(proceed (!sigmoid (randn `(10 10))))

{CPUTENSOR[float] :shape (10 10) :named ChainTMP13408 
  :vec-state [computed]
  ((0.7712369  0.08284816 0.43262917 ~ 0.23829551 0.3080795  0.6284006)                  
   (0.32359025 0.3921803  0.3027691  ~ 0.44983754 0.88687694 0.27936012)   
               ...
   (0.4514993  0.6484144  0.4318265  ~ 0.23674174 0.7804215  0.8317024)
   (0.6750185  0.5297069  0.2650835  ~ 0.42000595 0.52054065 0.45882672))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] L1Norm

```
(L1Norm x p &key (:reduction :mean))
```

Returns a tensor that measures L1 Norm between each element in the input `x` and `y`.

```math
l(x, y) = L = {l_1, ..., l_n}^\intercal, l_n = abs(x_n - y_n)
```

where `N` is a batch-size.

In addition, reading the value of a `:reduction` keyword (one of `:mean` `:sum` `nil`), the result of `L` is reducted. (If nil, reduction is ignored.)

### Example

```lisp
(proceed (L1Norm (randn `(10 10)) (randn `(10 10))))

{CPUTENSOR[float] :shape (1 1) -> :view (<(BROADCAST 1)> <(BROADCAST 1)>) -> :visible-shape (1 1) :named ChainTMP13824 
  :vec-state [computed]
  ((1.148601))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] mse
```
(mse x p &key (:reduction :mean))
```
Returns a tensor that measures the MSE error (L2Norm) between each element in the input `x` and `y`.

```math
l(x, y) = L = {l_1, ..., l_n}^\intercal, l_n = (x_n - y_n)^2
```

where `N` is a batch-size.

In addition, reading the value of a `:reduction` keyword (one of `:mean` `:sum` `nil`), the result of `L` is reducted. (If nil, reduction is ignored.)

### Example

```lisp
(proceed (MSE (randn `(10 10)) (randn `(10 10))))

{CPUTENSOR[float] :shape (1 1) -> :view (<(BROADCAST 1)> <(BROADCAST 1)>) -> :visible-shape (1 1) :named ChainTMP14222 
  :vec-state [computed]
  ((2.182282))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [fucntion] cross-entropy-loss

```lisp
(cross-entropy-loss x labels &key (delta 1e-7) (reduction :mean))
```

Returns a tensor that measures the Cross-Entropy-Error between each element in the x and labels.

```math
L_i = -p_ilog(x_i + delta)
```

```math
\begin{equation}
  out_i=
  \begin{cases}
    sum(L)  & \text{reduction = sum} \\
    mean(L) & \text{reduction = mean} \\
    L       & \text{otherwise}
  \end{cases}
\end{equation}
```

### Inputs

`x[AbstractTensor]`

`labels[AbstractTensor]` one-hot encoding.

## [function] softmax-cross-entropy

```lisp
(softmax-cross-entropy x labels)
```

Returns a tensor that measures the Softmax-Cross-Entropy-Error between each element in the x and labels.

```math
out = CrossEntropyLoss(Softmax(x), labels)
```

### Inputs

`x[AbstractTensor]`

`labels[AbstractTensor]` one-hot encoding.

## [model] LINEARLAYER

```
(linearlayer IN-FEATURES OUT-FEATURES &OPTIONAL (USE-BIAS? T))
```


which transformation of shapes are defined as:
```
([~ BATCH-SIZE IN-FEATURES] -> [~ BATCH-SIZE OUT-FEATURES])
```
### Description

Applies a linear transformation to the incoming data.

```math
y = xA^\intercal + b
```

### Inputs

`in-features[fixnum]` size of each input size.

`out-features[fixnum]` size of each output size.

`bias[boolean]` If set to nil, the layer won't learn an additive bias. default: `t`

### Parameters

`(linear-weight self)` the trainable value of the model of shape `out_features * in_features`. The values are sampled from  `xavier-uniform`.

`(linear-bias self)` the tranable value of bias of shape `out_features`. If bias is t, the initial values are sampled from uniform distribution: `U(-k, k)` where k = `sqrt(1/out-features)`.


### Example

```lisp
(LinearLayer 10 5)

<Composite: LINEARLAYER{W14226}(
    <Input : ((~ BATCH-SIZE 10)) -> Output: ((~ BATCH-SIZE 5))>

    WEIGHTS -> (5 10)
    BIAS    -> (5)
)>
```

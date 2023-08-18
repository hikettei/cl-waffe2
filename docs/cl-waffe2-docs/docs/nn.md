
# cl-waffe2/nn

## [function] !relu

Returns a tensor that applied ReLU element-wise.

```math
ReLU(x) = max(x, 0)
```
### Example

```lisp
(proceed (!relu (randn `(10 10))))

{CPUTENSOR[float] :shape (10 10) :named ChainTMP974 
  :vec-state [computed]
  ((-0.0         0.33708817   0.8596226    ~ 0.44115165   -0.0         0.4445048)                    
   (-0.0         0.89838934   1.0843782    ~ -0.0         -0.0         -0.0)   
                 ...
   (0.36703286   -0.0         1.7522626    ~ -0.0         -0.0         -0.0)
   (-0.0         2.0839996    -0.0         ~ 1.1435975    -0.0         -0.0))
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

{CPUTENSOR[float] :shape (10 10) :named ChainTMP1242 
  :vec-state [computed]
  ((0.19734775 0.78826183 0.5115942  ~ 0.28593835 0.25547844 0.86766356)                  
   (0.728437   0.631219   0.62524664 ~ 0.1444213  0.4615509  0.45974594)   
               ...
   (0.71474916 0.5147265  0.43981186 ~ 0.52759    0.7727495  0.85379267)
   (0.5236434  0.69857574 0.09652317 ~ 0.51632696 0.80563885 0.53765744))
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

{CPUTENSOR[float] :shape (1 1) -> :view (<(BROADCAST 1)> <(BROADCAST 1)>) -> :visible-shape (1 1) :named ChainTMP1611 
  :vec-state [computed]
  ((1.2166054))
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

{CPUTENSOR[float] :shape (1 1) -> :view (<(BROADCAST 1)> <(BROADCAST 1)>) -> :visible-shape (1 1) :named ChainTMP1928 
  :vec-state [computed]
  ((1.5432957))
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

<Composite: LINEARLAYER{W1935}(
    <Input : ((~ BATCH-SIZE 10)) -> Output: ((~ BATCH-SIZE 5))>

    WEIGHTS -> (5 10)
    BIAS    -> (5)
)>
```

## [model] CONV2D

```
(conv2d IN-CHANNELS OUT-CHANNELS KERNEL-SIZE &KEY (STRIDE 1) (PADDING 0) (DILATION 1) (GROUPS
                                                                                1) (BIAS
                                                                                    T) &AUX (STRIDE
                                                                                             (MAYBE-TUPLE
                                                                                              STRIDE
                                                                                              'STRIDE)) (KERNEL-SIZE
                                                                                                         (MAYBE-TUPLE
                                                                                                          KERNEL-SIZE
                                                                                                          'KERNEL-SIZE)) (PADDING
                                                                                                                          (MAYBE-TUPLE
                                                                                                                           PADDING
                                                                                                                           'PADDING)) (DILATION
                                                                                                                                       (MAYBE-TUPLE
                                                                                                                                        DILATION
                                                                                                                                        'DILATION)))
```


which transformation of shapes are defined as:
```
(INPUT[N C_IN H_IN W_IN] -> OUTPUT[N C_OUT H_OUT W_OUT] WHERE C_IN =
 IN-CHANNELS C_OUT = OUT-CHANNELS H_OUT =
 (IF (NUMBERP H_IN)
     (FLOOR
      (+ 1
         (/
          (+ H_IN (* 2 (CAR PADDING))
             (* (- (CAR DILATION)) (- (CAR KERNEL-SIZE) 1)) -1)
          (CAR STRIDE))))
     -1)
 W_OUT =
 (IF (NUMBERP W_IN)
     (FLOOR
      (+ 1
         (/
          (+ W_IN (* 2 (SECOND PADDING))
             (* (- (SECOND DILATION)) (- (SECOND KERNEL-SIZE) 1)) -1)
          (SECOND STRIDE))))
     -1))
```
### Description


Applies a 2D convolution over an input signal composed of several input planes.


### Inputs

`in-channels[fixnum]` `out-channels[fixnum]` the number of channels. For example, if the input image is RGB, `in-channels=3`.

`kernel-size[list (kernel-x kernel-y)]` controls the size of kernel (e.g.: `'(3 3)`).

`padding[fixnum or list]` controls the amount of padding applies to the coming input. pads in X and Y direction when an integer value is entered. set a list of `(pad-x pad-y)` and pads in each direction.

`stride[fixnum or list]` controls the stride for cross-correlation. As with `padding`, this parameter can be applied for each x/y axis.

`dilation[fixnum or list]` controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. (currently not working, please set 1.)

`bias[boolean]` Set t to use bias.

### Parameters

Let k be `(/ groups (* in-channels (apply #'* kernel-size)))`.


`weight` is a trainable parameter of `(out-channels, in-channels / groups, kernel-size[0] kernel-size[1])` sampled from `U(-sqrt(k), sqrt(k))` distribution, and can be accessed with `weight-of`.

`bias` is a trainable parameter of `(out-channels)` sampled from `U(-sqrt(k), sqrt(k))` distribution, and can be accessed with `bias-of`.

Note: When `Conv2D` is initialised, the output is displayed as -1. This is because the calculation of the output is complicated and has been omitted. Once call is invoked, the output is recorded.


### Example

```lisp
(Conv2D 3 5 '(3 3))

<Composite: CONV2D{W1945}(
    <Input : ((N 3 H_IN W_IN)) -> Output: ((N 5 -1 -1))>

    WEIGHT -> (5 3 3 3)
    BIAS   -> (5)
)>
```

## [model] MAXPOOL2D

```
(maxpool2d KERNEL-SIZE &KEY (STRIDE KERNEL-SIZE) (PADDING 0) &AUX (STRIDE
                                                         (MAYBE-TUPLE STRIDE
                                                                      'STRIDE)) (PADDING
                                                                                 (MAYBE-TUPLE
                                                                                  PADDING
                                                                                  'PADDING)))
```


which transformation of shapes are defined as:
```
(INPUT[N C H_IN W_IN] -> OUTPUT[N C H_OUT W_OUT] WHERE H_OUT =
 (IF (NUMBERP H_IN)
     (POOL-OUT-SIZE H_IN (CAR PADDING) (CAR KERNEL-SIZE) (CAR STRIDE))
     -1)
 W_OUT =
 (IF (NUMBERP W_OUT)
     (POOL-OUT-SIZE W_IN (SECOND PADDING) (SECOND KERNEL-SIZE) (SECOND STRIDE))
     -1))
```
### Description


Applies a 2D max pooling over an input signal composed of several input planes.

### Inputs

`kernel-size[list]` the size of window

`stride[fixnum or list]` the stride of window

`padding[fixnum or list]` adds 0 padding

Likewise `Conv2D`, these parameters can be set for both X and Y axis directions.


## [model] AVGPOOL2D

```
(avgpool2d KERNEL-SIZE &KEY (STRIDE KERNEL-SIZE) (PADDING 0) &AUX (STRIDE
                                                         (MAYBE-TUPLE STRIDE
                                                                      'STRIDE)) (PADDING
                                                                                 (MAYBE-TUPLE
                                                                                  PADDING
                                                                                  'PADDING)))
```


which transformation of shapes are defined as:
```
(INPUT[N C H_IN W_IN] -> OUTPUT[N C H_OUT W_OUT] WHERE H_OUT =
 (IF (NUMBERP H_IN)
     (POOL-OUT-SIZE H_IN (CAR PADDING) (CAR KERNEL-SIZE) (CAR STRIDE))
     -1)
 W_OUT =
 (IF (NUMBERP W_OUT)
     (POOL-OUT-SIZE W_IN (SECOND PADDING) (SECOND KERNEL-SIZE) (SECOND STRIDE))
     -1))
```
### Description


Applies a 2D average pooling over an input signal composed of several input planes.

### Inputs

`kernel-size[list]` the size of window

`stride[fixnum or list]` the stride of window

`padding[fixnum or list]` adds 0 padding

Likewise `Conv2D`, these parameters can be set for both X and Y axis directions.


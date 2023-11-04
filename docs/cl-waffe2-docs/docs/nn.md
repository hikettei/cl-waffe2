
# cl-waffe2/nn
## [Non Linear Activations]

## [function] !relu

```lisp
(!relu x)
```

Computes ReLU to the given tensor.

```math
ReLU(x) = max(x, 0)
```

### Example

```lisp
(proceed (!relu (randn `(10 10))))

{CPUTENSOR[float] :shape (10 10) :id TID1943 
  :vec-state [computed]
  ((-0.0        -0.0        -0.0        ~ 1.6821892   0.36130282  -0.0)                   
   (0.87504226  1.429912    -0.0        ~ -0.0        -0.0        -0.0)   
                ...
   (-0.0        -0.0        0.2992248   ~ 0.6288745   0.5963682   0.55514115)
   (-0.0        -0.0        -0.0        ~ -0.0        -0.0        0.7646121))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] !gelu

```lisp
(!gelu x)
```

Applies the Gaussian Error Linear Units function approximated with:

```math
GeLU(x) = 0.5\times{x}\times{(1 + Tanh(\sqrt{\frac{2}{π}}\times{(x + 0.44715\times{x^3})}))}
```


### Example

```lisp
(proceed (!relu (randn `(10 10))))

{CPUTENSOR[float] :shape (10 10) :id TID1987 
  :vec-state [computed]
  ((-0.0        1.9134964   1.9286379   ~ 0.9884213   -0.0        1.4884331)                   
   (-0.0        -0.0        0.1880067   ~ 1.2277454   -0.0        0.3174525)   
                ...
   (0.5903688   0.32395032  -0.0        ~ -0.0        -0.0        -0.0)
   (0.42296785  -0.0        -0.0        ~ -0.0        0.3376901   0.36640626))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] !sigmoid

```lisp
(!sigmoid x)
```

Computes sigmoid function to the given tensor.

```math
Sigmoid(x) = \frac{1}{1 + exp(-x)}
```

### Example

```lisp
(proceed (!sigmoid (randn `(10 10))))

{CPUTENSOR[float] :shape (10 10) :id TID2027 
  :vec-state [computed]
  ((0.86408615  0.6105685   0.3788395   ~ 0.25450796  0.7772302   0.67822224)                   
   (0.8417765   0.5603245   0.36006212  ~ 0.7668568   0.50954837  0.24739715)   
                ...
   (0.5413329   0.2221309   0.36529404  ~ 0.32194504  0.49215865  0.5152154)
   (0.59982526  0.9218087   0.6415686   ~ 0.5698819   0.4574838   0.40995663))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] !leakey-relu

```lisp
(!leakey-relu x &key (negative-slope 0.01))
```

Applies the element-wise function:

```math
LeakeyReLU(x) = max(x, 0) + negative-slope\times{min(0, x)}
```

### Inputs

`x[AbstractTensor]`

`negative-slope[single-float]`

### Example

```lisp
(proceed (!leakey-relu (randn `(10 10))))

{CPUTENSOR[float] :shape (10 10) :id TID2146 
  :vec-state [computed]
  ((0.78715545    0.102183476   -0.0035984537 ~ 0.5117078     0.10139906    0.22480665)                     
   (-0.008864752  0.37595636    -0.0015632007 ~ 0.42575783    2.48219       0.042828444)   
                  ...
   (0.37137523    -0.001629176  -0.0038587882 ~ 0.19631904    0.71814036    -0.0044727987)
   (-0.007339712  0.49409094    -0.010092152  ~ -0.004170165  1.1420414     0.45225763))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] !elu

```lisp
(!elu x &key (alpha 1.0))
```

Applies the Expotential Linear Units Function (ELUs) element-wise as described in [this paper](https://arxiv.org/abs/1511.07289)

```math
\begin{equation}
  ELU(x)=
  \begin{cases}
    \text{x} & if x>0 \\
    \text{α*(exp(x)-1)} & \text{otherwise}
  \end{cases}
\end{equation}
```

### Example

```lisp
(proceed (!leakey-relu (randn `(10 10))))

{CPUTENSOR[float] :shape (10 10) :id TID2190 
  :vec-state [computed]
  ((1.2949203     1.319317      -0.0013560241 ~ 0.35411915    0.13482997    -0.01092499)                     
   (2.6539629     -0.0019639763 -0.008898432  ~ 0.5323376     -0.0083555095 -0.016043724)   
                  ...
   (0.5872035     1.3011793     -1.4846033e-4 ~ -0.018327065  0.5999392     0.111192465)
   (-0.006346898  1.8144846     -0.006539148  ~ 0.60142326    -0.00977075   0.9043215))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] !softmax

```lisp
(!softmax x &key (avoid-overflow t) (axis 1))
```

Returns a tensor that applied Softmax function along the given axis.

```lisp
Softmax(x_i) = exp(x_i)\div{sum(x_j, axis)}
```

If avoid-overflow is set to t:

```lisp
x_i = x_i - mean(x)
```

### Inputs

`avoid-overflow[boolean]` If t, `exp(x_i)` is substracted by the mean value of `x`.

`axis[fixnum or list or t]` The axis to be reducted.

### Example

```lisp
(proceed (!softmax (randn `(3 3))))

{CPUTENSOR[float] :shape (3 3) :id TID2291 
  :vec-state [computed]
  ((0.4770026   0.42076913  0.10222831)
   (0.074501365 0.38638794  0.53911066)
   (0.43734843  0.06926617  0.49338531))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```
## [Normalization Layers]

## [model] BATCHNORM2D

```
(batchnorm2d IN-FEATURES &KEY (AFFINE T) (EPS 1.0e-5))
```


which transformation of shapes are defined as:
```
(X[N IN-FEATURES H W] -> OUT[N IN-FEATURES H W])
```
### Description

Applies Batch Normalization over a 4D input (N C H W) as described in the paper [Batch Normalization](https://arxiv.org/abs/1502.03167).

```math
BatchNorm(x) = \frac{x - E[x]}{\sqrt{Var[x] + ε}}\times{γ}+β
```

### Inputs

`in-features[fixnum]` - C from an expected input size (N C H W)

`affine[bool]` Set T to apply affine transofmration to the output. In default, set to t.

`eps[single-float]` a value added to the denominator for numerical stability. Default: 1e-5.

### Parameters

`alpha` (in-features) is a trainable tensor filled with `1.0`. accessor: `alpha-of`

`beta`  (in-features) is a trainable tensor filled with `0.0`. accessor: `beta-of`


## [model] LAYERNORM2D

```
(layernorm2d NORMALIZED-SHAPE &KEY (EPS 1.0e-5) (AFFINE T))
```


which transformation of shapes are defined as:
```
(X[~ NORMALIZED-SHAPE] -> OUT[~ NORMALIZED-SHAPE])
```
### Description

Applies Layer Normalization over a mini-batch of inputs as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450)

```math
LayerNorm(x) = \frac{x - E[x]}{\sqrt{Var[x] + ε}}\times{γ}+β
```

The mean and standard-deviation are calculated over the last D dimension where D = `(length normalized-shape)`. The parameters β and γ are trainable affine transforms created if `affine` is set to T.

### Inputs

`normalized-shape` [list or fixnum] the size of kernel

`eps[single-float]` a value added to the denominator for the numerical stability.

`affine[boolean]` Set T to use affine transformation.

### Parameters

`alpha` (normalized-shape) is a trainable tensor filled with `1.0`. accessor: `alpha-of`

`beta` (normalized-shape) is a trainable tensor filled with `0.0`. accessor: `beta-of`

## [Loss Functions]

### Tips: Utility Function

The `:reduction` keyword for all loss functions is set to T by default. If you want to compose several functions for reduction (e.g. ->scal and !sum), it is recommended to define utilities such as:

```lisp
(defun criterion (criterion X Y &key (reductions nil))
  (apply #'call->
	 (funcall criterion X Y)
	 (map 'list #'asnode reductions)))

;; Without criterion:
(->scal (MSE x y :reduction :sum))

;; With criterion for example:
(criterion #'MSE x y :reductions `(#'!sum #'->scal))
```

## [function] L1Norm

```
(L1Norm x p &key (:reduction t))
```

Returns a tensor that measures L1 Norm between each element in the input `x` and `y`.

```math
l(x, y) = L = {l_1, ..., l_n}^\intercal, l_n = abs(x_n - y_n)
```

where `N` is a batch-size.

In addition, reading the value of a `:reduction` keyword (one of `:mean` `:sum` `t`), the result of `L` is reducted. (If t, reduction is ignored.)

### Example

```lisp
(proceed (L1Norm (randn `(10 10)) (randn `(10 10))))

{CPUTENSOR[float] :shape (10 10) :id TID2395 
  :vec-state [computed]
  ((0.21944374   3.4906983    0.09639442   ~ 1.7210886    1.748786     0.537158)                    
   (0.0011370182 0.8881786    2.1463056    ~ 2.0200849    0.9273455    1.6265762)   
                 ...
   (0.96517265   1.3054338    1.6356181    ~ 0.64939415   0.15627737   0.7375276)
   (0.28122553   1.8060057    1.0688223    ~ 0.42294133   1.3473802    1.7486184))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] mse
```
(mse x p &key (:reduction T))
```

Returns a tensor that measures the MSE error (i.e.: L2Norm) between each element in the input `x` and `y`.

```math
l(x, y) = L = {l_1, ..., l_n}^\intercal, l_n = (x_n - y_n)^2
```

where `N` is a batch-size.

In addition, reading the value of a `:reduction` keyword (one of `:mean` `:sum` `t`), the result of `L` is reducted. (If t, this operation is ignored.)

### Example

```lisp
(proceed (MSE (randn `(10 10)) (randn `(10 10))))

{CPUTENSOR[float] :shape (10 10) :id TID2450 
  :vec-state [computed]
  ((18.540094    0.010009353  5.0938554    ~ 10.227004    0.20378923   0.039667178)                    
   (0.16211173   2.0781722    2.433506     ~ 5.156483     0.10430828   1.3060123)   
                 ...
   (6.0381346    0.30820525   0.018570788  ~ 2.4814055    1.2736448    0.054688945)
   (0.1717851    0.7182333    0.6443218    ~ 0.20784199   0.03354252   1.3616705))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [fucntion] cross-entropy-loss

```lisp
(cross-entropy-loss x labels &key (delta 1e-7) (reduction t))
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

`reduction` one of :sum :mean t

## [function] softmax-cross-entropy

```lisp
(softmax-cross-entropy x labels &key (axis 1) (delta 1e-7) (avoid-overflow nil) (reduction t))
```

Returns a tensor that measures the Softmax-Cross-Entropy-Error between each element in the x and labels.

```math
out = CrossEntropyLoss(Softmax(x), labels)
```

### Inputs

`x[AbstractTensor]` distribution to measure

`labels[AbstractTensor]` answer labels with one-hot encoding.
## [Linear Layers]

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

`(linear-bias self)` the trainable value of bias of shape `out_features`. If bias is t, the initial values are sampled from a uniform distribution: `U(-k, k)` where k = `sqrt(1/out-features)`.


### Example

```lisp
(LinearLayer 10 5)

<Composite: LINEARLAYER{W2500}(
    <Input : ((~ BATCH-SIZE 10)) -> Output: ((~ BATCH-SIZE 5))>

    WEIGHTS -> (5 10)
    BIAS    -> (5)
)>
```
## [Dropout Layers]
## [Sparse Layers]
## [Recurrent Layers]
## [Convolutional Layers]

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
 (CONV-OUT-SIZE H_IN (CAR PADDING) (CAR DILATION) (CAR KERNEL-SIZE)
  (CAR STRIDE))
 W_OUT =
 (CONV-OUT-SIZE W_IN (SECOND PADDING) (SECOND DILATION) (SECOND KERNEL-SIZE)
  (SECOND STRIDE)))
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

<Composite: CONV2D{W2510}(
    <Input : ((N 3 H_IN W_IN)) -> Output: ((N 5
                                            [observe-axis({[(CONV-OUT-SIZE H_IN (CAR PADDING) (CAR DILATION) (CAR KERNEL-SIZE)
                 (CAR STRIDE))]}, H_IN, (0 0), (1 1), (3 3), (1 1))]
                                            [observe-axis({[(CONV-OUT-SIZE W_IN (SECOND PADDING) (SECOND DILATION) (SECOND KERNEL-SIZE)
                 (SECOND STRIDE))]}, W_IN, (0 0), (1 1), (3 3), (1 1))]))>

    WEIGHT -> (5 3 3 3)
    BIAS   -> (5)
)>
```
## Pooling Layers

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
 (POOL-OUT-SIZE H_IN (CAR PADDING) (CAR KERNEL-SIZE) (CAR STRIDE)) W_OUT =
 (POOL-OUT-SIZE W_IN (SECOND PADDING) (SECOND KERNEL-SIZE) (SECOND STRIDE)))
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
 (POOL-OUT-SIZE H_IN (CAR PADDING) (CAR KERNEL-SIZE) (CAR STRIDE)) W_OUT =
 (POOL-OUT-SIZE W_IN (SECOND PADDING) (SECOND KERNEL-SIZE) (SECOND STRIDE)))
```
### Description


Applies a 2D average pooling over an input signal composed of several input planes.

### Inputs

`kernel-size[list]` the size of window

`stride[fixnum or list]` the stride of window

`padding[fixnum or list]` adds 0 padding

Likewise `Conv2D`, these parameters can be set for both X and Y axis directions.


## [function] unfold

```lisp
(unfold input dilation kernel-size stride padding)
```

Extracts sliding local blocks from a batched input tensor. The detailed specifications follow PyTorch: [nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html).

As of this writing, `input` must be a 4D Tensor even when `N=batch-size=1`.

Corresponding nodes: `cl-waffe2/base-impl:Im2ColNode`, `cl-waffe2/base-impl:Col2ImNode`

### Inputs

Note that `dilation`, `kernel-size`, `stride`, and `padding` are given in this form:

`(list y-direction(Height) x-direction(Width))`

`input[AbstractTensor]` the tensor to be unfold.

`dilation[list]` a parameter that controls the stride of elements within the neighborhood.

`kernel-size[list]` the size of sliding blocks.

`padding[list]` implicts the number of zero-padding to be added on both sides of input.

`stride[list]` the number of stride of the sliding blocks.

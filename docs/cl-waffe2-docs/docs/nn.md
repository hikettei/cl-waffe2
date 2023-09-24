
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

{CPUTENSOR[float] :shape (10 10) :id TID1503 
  :vec-state [computed]
  ((0.003321576 0.093405314 0.43108767  ~ 0.50773376  -0.0        0.09826224)                   
   (0.67178464  0.2550704   0.3619366   ~ 0.9756666   -0.0        -0.0)   
                ...
   (0.67445344  0.14898217  1.0800804   ~ 1.1706256   -0.0        -0.0)
   (-0.0        0.004279087 2.8531516   ~ -0.0        0.31110105  -0.0))
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

{CPUTENSOR[float] :shape (10 10) :id TID1594 
  :vec-state [computed]
  ((-0.0        0.970088    -0.0        ~ 2.643538    0.81806624  0.46306393)                   
   (0.21459135  2.1423104   -0.0        ~ -0.0        1.621045    -0.0)   
                ...
   (1.711752    0.24372405  0.93948185  ~ -0.0        -0.0        1.4903362)
   (0.11486413  0.42094356  -0.0        ~ -0.0        0.11685378  0.78753096))
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

{CPUTENSOR[float] :shape (10 10) :id TID1681 
  :vec-state [computed]
  ((0.5256962   0.7378315   0.1961688   ~ 0.7411308   0.6399446   0.62830925)                   
   (0.5077358   0.89164513  0.58502465  ~ 0.59738374  0.4095264   0.54009235)   
                ...
   (0.80688757  0.27547944  0.25196192  ~ 0.31511673  0.40611273  0.40410277)
   (0.28302354  0.73229575  0.55372566  ~ 0.3960918   0.4791207   0.682953))
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

{CPUTENSOR[float] :shape (10 10) :id TID1837 
  :vec-state [computed]
  ((-0.010987854  -0.0037596084 0.11885163    ~ -0.008229563  2.2673395     -0.0044549606)                     
   (-0.012962578  0.121423334   -0.006803466  ~ -0.012293068  1.1865982     1.3988262)   
                  ...
   (-0.018473854  0.1880067     1.3290814     ~ -0.0070048855 0.3174525     -0.009075811)
   (1.2731968     -0.011155452  -1.3661921e-5 ~ 0.58505344    -5.429155e-4  -0.0032047757))
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

{CPUTENSOR[float] :shape (10 10) :id TID1928 
  :vec-state [computed]
  ((0.13979843    0.5125472     -0.019835515  ~ 0.31957033    -0.0048134336 -0.011082122)                     
   (-0.0012673637 -0.009207103  -0.014301519  ~ 0.6066013     1.1659489     -0.0024188054)   
                  ...
   (0.242479      -0.005750944  0.41514573    ~ 0.03819821    -0.011125428  1.8616861)
   (-0.0058236853 -0.0016050143 -0.0034687081 ~ -0.0052875406 0.462297      -0.02770405))
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

{CPUTENSOR[float] :shape (3 3) :id TID2094 
  :vec-state [computed]
  ((0.43811542 0.3959875  0.16589706)
   (0.3660144  0.546256   0.08772959)
   (0.2143596  0.11867718 0.6669633))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```
## [Normalization Layers]

## [model] LAYERNORM

```
(layernorm NORMALIZED-SHAPE &KEY (EPS 1.0e-5) (AFFINE T))
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

{CPUTENSOR[float] :shape (10 10) :id TID2233 
  :vec-state [computed]
  ((1.5507355   0.7457546   1.0513756   ~ 1.1928346   0.57043034  0.5599009)                   
   (0.27099022  0.44716904  0.041606337 ~ 0.23732874  1.4682146   0.70283914)   
                ...
   (1.247272    0.7221159   0.24704999  ~ 1.3999897   0.38727573  0.07190052)
   (1.2126788   0.17113328  0.78247666  ~ 0.51140684  1.8648069   2.7227192))
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

{CPUTENSOR[float] :shape (10 10) :id TID2335 
  :vec-state [computed]
  ((0.674522     0.14472504   0.8048076    ~ 0.11429957   0.30557308   2.6626682)                    
   (3.6498027    1.0817418    0.661556     ~ 0.12610506   0.13591279   0.043855775)   
                 ...
   (0.7888612    4.6066275    1.3160336    ~ 0.8599697    2.64575      6.1790547)
   (0.24884623   0.116668575  0.1191209    ~ 6.7462487    2.4428065    1.3009849))
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

`(linear-bias self)` the tranable value of bias of shape `out_features`. If bias is t, the initial values are sampled from uniform distribution: `U(-k, k)` where k = `sqrt(1/out-features)`.


### Example

```lisp
(LinearLayer 10 5)

<Composite: LINEARLAYER{W2432}(
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

<Composite: CONV2D{W2442}(
    <Input : ((N 3 H_IN W_IN)) -> Output: ((N 5 -1 -1))>

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

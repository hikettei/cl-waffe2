
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

{LISPTENSOR[float] :shape (10 10) :id TID1863 
  :vec-state [computed]
  ((-0.0        0.39568463  -0.0        ~ -0.0        1.32649     -0.0)                   
   (0.8187249   0.33749792  2.6600037   ~ 0.93948185  2.7282104   0.947536)   
                ...
   (0.55305374  -0.0        -0.0        ~ 0.06831199  -0.0        -0.0)
   (-0.0        0.21929178  0.9859774   ~ -0.0        -0.0        0.9216246))
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

{LISPTENSOR[float] :shape (10 10) :id TID1909 
  :vec-state [computed]
  ((1.4845713   0.34032807  0.48264125  ~ 0.71006465  -0.0        0.032608878)                   
   (0.72863156  1.6821892   0.36130282  ~ -0.0        0.41997707  -0.0)   
                ...
   (-0.0        0.93430895  -0.0        ~ 1.7637728   0.5941059   0.7955804)
   (-0.0        0.6288745   0.5963682   ~ 2.32079     -0.0        1.9969639))
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

{LISPTENSOR[float] :shape (10 10) :id TID2003 
  :vec-state [computed]
  ((0.51031536 0.3184848  0.31827003 ~ 0.11868909 0.6164283  0.7810872)                  
   (0.19540769 0.728776   0.48958635 ~ 0.79068863 0.37906235 0.3462106)   
               ...
   (0.7547459  0.63190573 0.55480015 ~ 0.79401666 0.49140203 0.18795623)
   (0.21578257 0.3503381  0.34892797 ~ 0.37108573 0.3957934  0.75265163))
  :facet :input
  :belongs-to :memory-pool
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

## [function] !leaky-relu

```lisp
(!leakey-relu x &key (negative-slope 0.01))
```

Applies the element-wise function:

```math
LeakyReLU(x) = max(x, 0) + negative-slope\times{min(0, x)}
```

### Inputs

`x[AbstractTensor]`

`negative-slope[single-float]`

### Example

```lisp
(proceed (!leaky-relu (randn `(10 10))))

{LISPTENSOR[float] :shape (10 10) :id TID2071 
  :vec-state [computed]
  ((0.09089017    -0.008739611  0.3376901     ~ 0.07416397    -0.00800783   0.41286528)                     
   (-0.0026475843 -0.0107471235 1.2495973     ~ 0.41514573    0.43301943    -0.009638567)   
                  ...
   (-0.0034302538 1.4131424     -8.48679e-4   ~ 0.24107653    -0.0036714186 0.36296296)
   (-0.008866108  -0.007448476  -3.1367975e-4 ~ 0.04403485    -0.0076020416 0.54067177))
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
(proceed (!elu (randn `(10 10))))

{LISPTENSOR[float] :shape (10 10) :id TID2105 
  :vec-state [computed]
  ((-0.7769077   0.2813693    -0.15673691  ~ -0.46348953  0.087048896  -0.6981429)                    
   (-0.47141248  0.5117078    0.10139906   ~ 1.190436     -0.7454948   -0.716177)   
                 ...
   (-0.7487925   1.4930589    0.033359565  ~ 0.0027869258 -0.56012493  0.40648147)
   (-0.6087958   0.19631904   0.71814036   ~ 0.91196746   0.59532636   0.09223112))
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

{LISPTENSOR[float] :shape (3 3) :id TID2299 
  :vec-state [computed]
  ((0.074893534 0.16076672  0.76433975)
   (0.17536336  0.40728894  0.41734767)
   (0.36528662  0.49214125  0.14257213))
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

{LISPTENSOR[float] :shape (10 10) :id TID2394 
  :vec-state [computed]
  ((1.4621267   0.39796215  1.7387664   ~ 0.39032784  0.94705695  1.8357267)                   
   (1.6068172   0.9255259   1.0957334   ~ 0.78427774  2.3096297   0.28167194)   
                ...
   (1.0189981   0.42509243  1.4098688   ~ 0.71015525  0.3436821   1.1238005)
   (0.7587658   1.1173241   0.22767365  ~ 2.0795705   1.2111005   0.12396826))
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

{LISPTENSOR[float] :shape (10 10) :id TID2447 
  :vec-state [computed]
  ((5.4938188    0.17887937   1.8154333    ~ 0.8552882    0.5842989    0.6212674)                    
   (0.0027494146 0.87471426   0.363951     ~ 3.7917376    0.0064691147 4.112536)   
                 ...
   (0.07106308   6.7640767    2.9721603    ~ 2.0878654    0.15314594   0.5941522)
   (0.04261892   4.294284     0.8050398    ~ 5.645695     0.14531337   0.44783399))
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

<Composite: LINEARLAYER{W2489}(
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

<Composite: CONV2D{W2499}(
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

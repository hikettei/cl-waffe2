
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

{CPUTENSOR[float] :shape (10 10) :id TID1432 
  :vec-state [computed]
  ((-0.0        0.24354811  1.452569    ~ -0.0        0.970088    -0.0)                   
   (-0.0        -0.0        0.6617144   ~ 0.21459135  2.1423104   -0.0)   
                ...
   (1.32649     -0.0        0.8187249   ~ 1.711752    0.24372405  0.93948185)
   (2.7282104   0.947536    -0.0        ~ 0.11486413  0.42094356  -0.0))
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

{CPUTENSOR[float] :shape (10 10) :id TID1530 
  :vec-state [computed]
  ((0.35961413  0.40555617  0.004845902 ~ 0.10287541  1.0347279   -0.0)                   
   (1.5624609   2.4698493   -0.0        ~ 0.03094567  2.107657    0.34343487)   
                ...
   (-0.0        0.032608878 0.72863156  ~ 1.429912    -0.0        -0.0)
   (0.41997707  -0.0        0.9028499   ~ -0.0        1.0063016   0.21573552))
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

{CPUTENSOR[float] :shape (10 10) :id TID1619 
  :vec-state [computed]
  ((0.40291837 0.685333   0.3184058  ~ 0.24996755 0.40710145 0.529678)                  
   (0.54561454 0.29956734 0.7501292  ~ 0.2147955  0.53031856 0.33618394)   
               ...
   (0.6164283  0.7810872  0.19540769 ~ 0.13618016 0.5468637  0.79068863)
   (0.37906235 0.3462106  0.33601856 ~ 0.78128946 0.24683852 0.49965847))
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

{CPUTENSOR[float] :shape (10 10) :id TID1790 
  :vec-state [computed]
  ((1.6553823     -0.005161251  0.3189371     ~ 0.13979843    0.5125472     -0.019835515)                     
   (-0.0015778928 0.21411653    -0.011229473  ~ -0.0012673637 -0.009207103  -0.014301519)   
                  ...
   (-0.00800783   0.41286528    -0.0026475843 ~ 0.242479      -0.005750944  0.41514573)
   (0.43301943    -0.009638567  0.18777691    ~ -0.0058236853 -0.0016050143 -0.0034687081))
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

{CPUTENSOR[float] :shape (10 10) :id TID1888 
  :vec-state [computed]
  ((0.9162214     -0.007410889  1.113944      ~ 0.59393775    0.49283808    -0.0037717714)                     
   (0.34324142    0.74365646    -0.01085172   ~ 1.1646757     0.5146315     0.17069556)   
                  ...
   (0.087048896   -0.011978014  -0.006375469  ~ 0.37595636    -0.0015632007 1.190436)
   (-0.013684339  -0.012594044  -0.0240032    ~ -0.0067122946 -0.0064485115 0.39409873))
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

{CPUTENSOR[float] :shape (3 3) :id TID2060 
  :vec-state [computed]
  ((0.311157    0.36777267  0.32107037)
   (0.09865908  0.53212994  0.369211)
   (0.2003907   0.68643314  0.113176145))
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

{CPUTENSOR[float] :shape (10 10) :id TID2209 
  :vec-state [computed]
  ((0.49038488   0.481458     0.64320445   ~ 1.29865      0.044514775  0.18740225)                    
   (3.1957962e-4 1.4896753    0.48115116   ~ 0.5323907    1.1251445    0.14569467)   
                 ...
   (0.94705695   1.8357267    1.6068172    ~ 1.4452353    0.9629061    0.78427774)
   (2.3096297    0.28167194   1.5302484    ~ 1.4889115    2.3150122    1.815999))
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

{CPUTENSOR[float] :shape (10 10) :id TID2315 
  :vec-state [computed]
  ((1.1257133    0.30800548   0.87197316   ~ 0.038676202  1.5365238    0.37470114)                    
   (0.38163102   4.462762     0.16467464   ~ 2.3436475    0.23491123   0.12598184)   
                 ...
   (0.5842989    0.6212674    0.0027494146 ~ 0.0019333913 0.21851705   3.7917376)
   (0.0064691147 4.112536     1.9115908    ~ 6.7250705    0.2316556    1.1607686e-4))
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

<Composite: LINEARLAYER{W2416}(
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

<Composite: CONV2D{W2426}(
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


# Distributions

## Sampling matrices from distribution
cl-waffe2 provides a package `:cl-waffe2/distributions` which is used to sample matrices from the distributions.
## Common Format to the APIs
All sampling functions are defined in the following format via `define-tensor-initializer` macro.

```(function-name shape [Optional Arguments] &rest args &keys &allow-other-keys)```

That is, arguments passed to the `make-tensor` function can also be passed directly to the initializer functions.

### Example

```lisp
(normal `(10 10) 0.0 1.0 :requires-grad t)

{CPUTENSOR[float] :shape (10 10)  
  ((-0.49381727  -0.4243577   -1.4530901   ~ 0.09241722   -1.8139324   0.33998385)                    
   (0.5221619    -1.0098034   0.8181761    ~ 0.034676224  0.31745166   0.8633344)   
                 ...
   (1.5360299    1.2005202    0.29415232   ~ -0.018849093 0.10730301   0.7192831)
   (-1.3815284   0.1739938    1.9703826    ~ -0.69942784  0.88236964   -0.6999107))
  :facet :exist
  :requires-grad T
  :backward NIL}
```

### Example

```lisp
(ax+b `(10 10) 1 0 :dtype :uint8)

{CPUTENSOR[uint8] :shape (10 10)  
  ((0  1  2  ~ 7  8  9)          
   (10 11 12 ~ 17 18 19)   
       ...
   (80 81 82 ~ 87 88 89)
   (90 91 92 ~ 97 98 99))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## define-tensor-initializer
```lisp
(define-tensor-initializer (function-name (&rest args) initializer-lambda document &key (keep-order? nil)))
```


`define-tensor-initializer` is a macro which is used to define a initializer function.


Initializer function is a function whose arguments follow this format:

    (function-name shape <Initializer's Arguments> &rest initargs &key &allow-other-keys)

Input:

    function-name - the function is defined after this argument

    args          - Initializer's Arguments

    initializer-lambda - A form to be expanded as the sampling function, which must return a function of #'(lambda (i) ...) where i is the index of element.

    keep-order? - set t if the index is needed to sampling matrices.

Example:

```lisp
(define-initializer-function
    uniform-random
    (upfrom below)
  (let ((upfrom (coerce upfrom (dtype->lisp-type (dtype tensor))))
	(below  (coerce below  (dtype->lisp-type (dtype tensor)))))
    #'(lambda (i)
	(declare (ignore i))
	(sample-uniform-random upfrom below)))
    "")

(uniform-random `(10 10) 0.1 0.3 :requires-grad t)

{CPUTENSOR[float] :shape (10 10)  
  ((0.13149574  0.15135926  0.1569588   ~ 0.103781514 0.20610212  0.19365484)                   
   (0.2638953   0.12672275  0.21630599  ~ 0.16542184  0.10228193  0.12928057)   
                ...
   (0.20429519  0.12252951  0.17538154  ~ 0.22072719  0.18642941  0.11027551)
   (0.14372297  0.11097031  0.25514898  ~ 0.28739202  0.18398522  0.15176433))
  :facet :exist
  :requires-grad T
  :backward NIL}
```

(Note that new tensor is binded to tensor, being used to determined dtype etc...)

## ax+b

```lisp
(ax+b shape a b &rest initargs &key &allow-other-keys)
```

The function ax+b is a family of initializer functions, and samples matrices from arithmetic progression.

```math
out_n = an + b
```

Inputs:

    a, b - Coefficients of the above formula.
### Example

```lisp
(ax+b `(3 3) 1.0 0.0)

{CPUTENSOR[float] :shape (3 3)  
  ((0.0 1.0 2.0)
   (3.0 4.0 5.0)
   (6.0 7.0 8.0))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## beta

```lisp
(beta shape alpha beta &rest initargs &key &allow-other-keys)
```


The function beta is a family of initializer functions, and sample matrices from beta distribution.

### Reference

1. Generating Beta Variates with Nonintegral Shape Parameters (R. C. H. Cheng University of Wales Institute of Science and Technology)


2. https://dl.acm.org/doi/pdf/10.1145/359460.359482


Note: My implementation is unstable, being occurs floating-overflow constantly..., especially when min(alpha, beta) < 1.0 (i.e.: beta-bc)
### Example

```lisp
(beta `(3 3) 5.0 1.0)

{CPUTENSOR[float] :shape (3 3)  
  ((0.845245   0.98823535 0.53992635)
   (0.7730661  0.7142378  0.84765464)
   (0.9686189  0.76956904 0.8549012))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## bernoulli

```lisp
(bernoulli shape p &rest initargs &key &allow-other-keys)
```
The bernoulli is a family of initializer functions, and samples matrices from bernoulli distribution.

### Inputs

p - Takes 1 with probability p and 0 with probalibity (1-p).
### Example

```lisp
(bernoulli `(3 3) 0.3)

{CPUTENSOR[float] :shape (3 3)  
  ((1.0 1.0 0.0)
   (0.0 1.0 0.0)
   (0.0 0.0 0.0))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## chisquare

```lisp
(chisquare shape df &rest initargs &key &allow-other-keys)
```
The function chisquare is a family of initializer functions, and samples matrices from chisquare distributions.

### Inputs

df - degree of freedom.

### References

 https://github.com/lvaruzza/cl-randist
### Example

```lisp
(chisquare `(3 3) 1.0)

{CPUTENSOR[float] :shape (3 3)  
  ((0.08651206  1.2326496   0.8483967)
   (0.011596402 0.4021547   0.027511619)
   (0.08436891  0.5057191   0.011141564))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## expotential

```lisp
(expotential shape &rest initargs &key &allow-other-keys)
```
The function expotential is a family of initializer functions, and samples the expotential distribution using ziggurat algorithm with table-size=256.


### References

1. https://andantesoft.hatenablog.com/entry/2023/04/30/183032

2. Marsaglia, G., & Tsang, W. W. (2000). The ziggurat method for generating random variables. Journal of statistical software.

3. https://marui.hatenablog.com/entry/2023/01/23/194507
### Example

```lisp
(expotential `(3 3))

{CPUTENSOR[float] :shape (3 3)  
  ((0.22541419 1.338656   0.14029047)
   (0.47960132 5.64849    0.29346126)
   (0.2325543  0.06230082 0.83188677))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## gamma

```lisp
(gamma shape k &rest initargs &key &allow-other-keys)
```
The function gamma is a family of initializer functions, and samples matrices from the gamma distribution.

### References

1. https://github.com/lvaruzza/cl-randist

### Example

```lisp
(gamma `(3 3) 1.0)

{CPUTENSOR[float] :shape (3 3)  
  ((0.60517055  0.0193314   1.8423601)
   (0.55326015  0.20066561  1.0735147)
   (1.0844046   0.115611516 2.299866))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## normal

```lisp
(normal shape mean stddev &rest initargs &key &allow-other-keys)
```
The function normal is a family of initializer functions, and samples matrices from normal distribution.


### Reference

1. https://github.com/lvaruzza/cl-randist (seems to create ziggurat table with size=128)


### Inputs

mean

stddev - Standard Deviation, σ.

### Example

```lisp
(normal `(3 3) 1.0 0.0)

{CPUTENSOR[float] :shape (3 3)  
  ((1.0 1.0 1.0)
   (1.0 1.0 1.0)
   (1.0 1.0 1.0))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## uniform-random

```lisp
(uniform-random shape upfrom below &rest initargs &key &allow-other-keys)
```
The function uniform-random is a family of initializer funtions, and samples matrices from uniform random distribution using Common Lisp's standard function, `(random arg)`.

Input:

    upfrom, below. Each elements of returned tensor is in the range of: `[upfrom, below)`
### Example

```lisp
(uniform-random `(3 3) 2 4)

{CPUTENSOR[float] :shape (3 3)  
  ((3.6584299 3.6804876 2.5902324)
   (3.003483  3.9570756 2.6778693)
   (2.3750212 3.4133909 2.549292))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## randn

```lisp
(randn shape &rest initargs &key &allow-other-keys)
```
The function randn is a family of initializer functions, and samples the gaussian distributions using ziggurat algorithm with table-size=256.


### References


1. https://andantesoft.hatenablog.com/entry/2023/04/30/183032

2. Marsaglia, G., & Tsang, W. W. (2000). The ziggurat method for generating random variables. Journal of statistical software.

3. https://marui.hatenablog.com/entry/2023/01/23/194507
### Example

```lisp
(randn `(3 3))

{CPUTENSOR[float] :shape (3 3)  
  ((0.31103113  -1.7451742  0.6599348)
   (0.05832184  -0.44232148 -0.81750864)
   (-0.496161   0.1307186   -0.22390747))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

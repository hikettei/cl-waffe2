
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
  ((0.3266808    -1.3054699   0.18602815   ~ -0.9140944   -0.22505726  -0.15569693)                    
   (0.3095909    0.077898845  -0.44244626  ~ 0.46269673   -0.40228054  -0.40816537)   
                 ...
   (-0.81813675  1.4225258    -0.08328378  ~ -0.9490488   -0.54520386  1.0669057)
   (-0.54442966  -0.06799416  -0.3919469   ~ -0.273578    1.1178145    -1.9768039))
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
  ((0.71240944 0.847264   0.69012284)
   (0.75272816 0.8312693  0.9785166)
   (0.80858713 0.95975447 0.6866892))
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
  ((0.0 0.0 1.0)
   (1.0 0.0 0.0)
   (1.0 0.0 0.0))
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
  ((1.444597    0.179507    0.2541821)
   (0.08026809  0.07310683  0.003460823)
   (0.082672    0.22727925  0.5728937))
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
  ((0.16602875  0.51840174  0.021845827)
   (0.6818311   6.2415676   0.29937285)
   (1.1426421   0.90104485  0.9713096))
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
  ((1.3301286   0.9113628   2.2936785)
   (1.458179    1.1904559   0.022091847)
   (1.6347893   0.15075241  0.35237762))
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
  ((2.7340446 3.542842  3.4739382)
   (3.3757188 2.502474  3.263076)
   (3.1209097 2.6771405 2.324503))
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
  ((1.1390121   1.9054949   0.40986034)
   (0.8395159   1.5175884   0.044068847)
   (-0.7726815  0.5808442   0.561935))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

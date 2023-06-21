
# Distributions

## Samples matrices from distribution
cl-waffe2 provides a package :cl-waffe2/distributions which is used to sample matrices from the distributions.
## Common Format to the APIs
All sampling functions are defined in the following format via `define-tensor-initializer` macro.

```(function-name shape [Optional Arguments] &rest args &keys &allow-other-keys)```

That is, arguments passed to the `make-tensor` function can also be passed directly to the initializer functions.

### Example

```lisp
(normal `(10 10) 0.0 1.0 :requires-grad t)

{CPUTENSOR[float] :shape (10 10)  
  ((-1.1777736  -1.5293252  -0.08038678 ~ -0.5230843  -1.9797146  0.15480807)                   
   (-1.6179706  0.37712237  -0.83498365 ~ 0.5407585   0.6986199   -2.3237712)   
                ...
   (-1.1100876  0.5754472   0.21646172  ~ -2.0047095  -1.1506671  -0.69261456)
   (-3.3963487  1.1185373   0.862867    ~ -0.8784946  1.5187143   -1.0303308))
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

## DEFINE-TENSOR-INITIALIZER
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

## AX+B

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

## BETA

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
  ((0.96710485 0.69845265 0.85011274)
   (0.44774783 0.95295686 0.92015046)
   (0.91509503 0.82753897 0.6857265))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## BERNOULLI

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
  ((0.0 0.0 0.0)
   (0.0 1.0 1.0)
   (0.0 1.0 1.0))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## CHISQUARE

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
  ((0.23907769   0.07092617   0.016861092)
   (3.2354553    4.265375     2.8736973)
   (0.10222152   1.0788939    0.0039725727))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## EXPOTENTIAL

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
  ((4.2645674  0.6785309  0.9014305)
   (1.0038087  2.7358596  1.4401376)
   (0.64674526 0.1801507  0.26139924))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## GAMMA

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
  ((1.0482972   1.2824489   2.1072867)
   (0.057502534 0.036654595 0.2094097)
   (0.18192601  0.81434125  2.3717813))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## NORMAL

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

## UNIFORM-RANDOM

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
  ((2.5383434 3.3838556 3.4856105)
   (2.4167972 2.420381  2.6341114)
   (2.6869664 3.764454  3.6075912))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

## RANDN

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
  ((-0.13091457 -0.30610606 -0.21164778)
   (0.8948386   -2.2751138  1.0176823)
   (0.17455088  0.1292598   0.03674699))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

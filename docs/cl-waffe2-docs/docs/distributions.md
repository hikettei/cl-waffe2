
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
  ((-2.039073    2.1363842    0.4856452    ~ 1.3490102    -0.750389    -0.5306339)                    
   (1.1781243    0.24081734   0.2659924    ~ -0.83129555  -1.0675454   -0.40438387)   
                 ...
   (-0.8311747   -0.4938327   -1.8763269   ~ -0.64829946  -0.4458745   -0.6646508)
   (-2.18276     0.7794045    -0.5697511   ~ -0.7563723   -0.12596954  -0.472876))
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
  ((0.8364021  0.8976208  0.99033064)
   (0.8865608  0.99239224 0.8848019)
   (0.7562379  0.97992814 0.7644238))
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
  ((1.0 0.0 1.0)
   (1.0 0.0 1.0)
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
  ((0.050981984  0.0045932024 1.2810057)
   (0.50474      0.73581934   0.28017002)
   (0.28480044   0.3380389    0.00970952))
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
  ((0.47353485  0.5842454   1.7064404)
   (0.85115165  0.53042275  0.031387772)
   (0.25639144  0.91021144  0.16937655))
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
  ((0.06623174  0.9723331   1.2478018)
   (0.103160195 1.1820387   0.6591184)
   (0.21935773  1.273132    0.96095526))
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
  ((2.1255891 3.5020173 2.346451)
   (3.1401393 3.01733   2.1304102)
   (3.8687963 2.4499538 3.027814))
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
  ((-1.4259218  0.5869355   2.229303)
   (0.5977518   -0.13385284 -0.2503691)
   (-0.41290623 0.35953173  -0.764124))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

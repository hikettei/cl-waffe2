
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
  ((-0.0930601   -0.7955518   -0.2320479   ~ -0.19919659  0.5053014    -0.20112497)                    
   (-0.28931206  -0.87701094  0.3625228    ~ -0.6312274   -2.3019962   0.36486906)   
                 ...
   (-1.5363529   0.28731748   -0.4586338   ~ -0.73600495  0.95712537   1.6522883)
   (1.2883565    0.91169834   2.869378     ~ 2.150223     0.7248882    -0.4818239))
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
  ((0.8391376  0.96829844 0.3757152)
   (0.9150053  0.7509756  0.9709161)
   (0.782447   0.89865106 0.96879745))
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
  ((1.0 0.0 0.0)
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
  ((0.0683025   1.0229962   0.023815382)
   (1.2645968   0.6006345   1.7994667)
   (0.36835933  0.25498798  0.80459654))
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
  ((0.8669353   0.24338455  0.121354826)
   (0.33946362  1.5045028   1.9900182)
   (0.481504    1.4580338   1.4063467))
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
  ((2.3013742   2.8152425   0.056881204)
   (0.5659466   2.1788025   1.27337)
   (0.09022631  0.3975919   1.6128166))
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
  ((3.4028106 2.2382534 2.8039198)
   (3.294498  2.771839  3.9309936)
   (3.9023423 3.4698777 3.7805085))
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
  ((1.9816247    1.8748245    -0.030450566)
   (0.68228406   -0.88974875  0.62135047)
   (-0.028420867 -0.89215803  0.20699267))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

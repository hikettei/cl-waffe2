
# AbstractTensor

## AbstractTensor

[class] AbstractTensor

The class AbstractTensor is a fundamental datatype of dealing with various kernel (e.g.: CPU, Metal, CUDA...).

The class provides the fundamental features following:
    1. Lazy-Evaluated Multi-Dimensional Matrix APIs, and accordingly stride APIs for column/row major.
    2. Multi-Dimensional Matrix Offsets (i.e.: View APIs).
    3. Recording What Functions were called in the previous computation. (To construct backward.)
    4. vec container
    5. Keep Gradients
    6. Input API
    7. Trace Informations for JIT to create well-optimized computation node.

Users can extend this class and define the brand-new Tensor's Dtype depending on their use.

See the examples to understand how this could be achieved at ./source/backends/lisp/tensor.lisp. or ./source/backends/cpu.

## tensor-vec

`(tensor-vec tensor)`

Accessing the pointer/array the tensor has. Not until tensor-vec is called, the new area isn't allocated.
## mref

`(mref tensor &rest subscripts)`

Read-only. Only used for printing the tensor.
Whether you cares about performance or not, this function shouldn't be used ignoring for printing tensors.
## vref

`(vref tensor index)`

vref is a generic-function to access tensor's vec.

Whether you cares about performance or not, this function shouldn't be used ignoring for printing tensors.

If you've created a new backend with having different ptr-type (can't be accessed by aref), only you have to do is to redefine vref.
## parameter
The function parameter computes all the previous nodes of the given tensor, returning the new tensor with requires-grad=t.

Example:

```lisp
(parameter (randn `(3 3)))
```
## `*no-grad*`
[parameter] `*no-grad*`

TODO: DOC
## with-no-grad
```lisp
(with-no-grad &body body)
```

Under this macro, all operations don't create any gradients.
## make-tensor
Refering a first-priority of  *using-backends* (that is, a car part) to know what kernel to use, the function make-tensor creates and allocate a new matrix.

Input:
    - shape-or-scalar (Any), set list (consisted of fixnum) here to create a matrix, otherwise the ScalarTensor is forcibly created.
    - requires-grad (Boolean) Set t to create gradient. (e.g.: the tensor is needed to be optimized.)
    - dtype (keyword) Set dtype you wanna use. See also: (Dtype API)
    - vec (Anything) If you wanna pass the make-instance to already-allocated matrix, use this parameter.
    - order (member :column :row) 
    - initial-element (Optional)

With regard to practical usage, the tutorials would be more helpful rather than this document.
## make-input
Referring a first-priority of *using-backend* (i.e.: car part), the function make-input creates a InputTensor.
WIth regard to practical usage, visit my tutorial.

Input:
    - Shape [list] Consisted of Fixnum or Symbol.
    - named [keyword]
    - dtype (as it is)
    - order (as it is)(TODO) -> View APIs etc...

(in-package :cl-waffe2.docs)

(with-page *nodes* "Formulate Neural Networks"
  (insert "The package `:cl-waffe2/vm.nodes` provides a fundamental system for building neural networks using `AbstractTensor`.

This package can be divided into three main parts.

1. Shaping APIs (`:where`)
2. defnode  (The smallest unit of differentiable operations)
3. defmodel (Operations consisted of defnode, and static functions)

")

  (macrolet ((with-doc (name type &body body)
	       `(with-section (format nil "~(~a~)" (symbol-name ,name))
		  (placedoc ,name ,type)
		  ,@body)))
    (with-section "Shaping API"
      (insert "

When defining an operation in cl-waffe2 with a `defnode` macro, the shape of the matrix used in the operation must also be defined in the `:where` keyword.

This is a Shaping API, and responsible for shape inspection of all operations.

## Introducing Subscript DSL

I assume you have already seen `defnode` macro. This macro takes a strange syntax language after :where keyword.

```lisp
(defnode (TransposeNode (myself)
            :where (A[~~ i j] -> A[~~ j i])
         ...))

(defnode (ScalarAdd (myself)
            :where (A[~~] Scal[scal] -> A[~~] where scal = 1)
        ...))

(defnode (ReshapeNode (myself tensor after &aux (before (shape tensor)))
            :where (A[before] -> A[after])
        ...))
```

This is a DSL (Domain Specific Language) called `Subscript DSL`, which is used to notate the pointer and shape to be handled before and after the operation.

For example, `TransposeNode` is said to be:

1. Before and after the operation, we use the same pointer.

2. A is a tensor with more than two dimensions, and after the operation, transposed the last two axes. (i.e.: A=(10 5 2), (10 2 5) is returned)

`ScalarAdd` is said to be:

1. The first argument `A` can be anything. The second argument `Scal` is a scalar tensor.

2. The returned tensor shares the pointer with the given `A`.

`ReshapeNode` is:

1. Before and after the operation, pointers are common.

2. The shape of A will be transformed from `before` into `after`

### Basic Grammar

Let's start with learning the grammar.

One line code of Subscript DSL follows this format:

```
[Before The Operation] -> [After The Operation] where [symbol = expression (Optional)] ...
```

Note that:

1. the pharse `where [symbol = expression (Optional)] ...` is **Optional**

2. One Subscript DSL place can include one line of code.

3. [Before The Operation] and [After The Operation] has the common grammar rule.

Let `<Arguments>` be a grammar rule of [Before The Operation] and [After The Operation], `<Arguments>` can be defined as:

```
<Arguments> ::= <Arguments> <Argument>
<Argument> ::= <PointerName> [ <SubScripts> ] | NIL

<PointerName> ::= Symbol // the same as CL's symbol.

<SubScripts>  ::= <Subscripts> <Subscript>
<Subscript>   ::= Symbol | NIL
```

To put it bluntly, <Argument> can be a sequence of:

```c
PointerName[SubScripts]

// SubScripts can be one of: [A], [A B] [~~ i j] etc...
```

### Assigned task

```
A[a b] B[a b] -> B[a b]
```

In the DSL above, `A` and `B` indicates the name of pointer, they're not needed to be defined in advance.

On the other hand `a` and `b` inside [ ... ], indicates subscripts of `A` and `B`, DSL's assigned work is to inference these **undetermined symbols** from:

1. determined symbol from `where` pharse and symbols in arguments of constructor.

2. Shape of the given inputs at runtime.

If any, DSL compiles and display a report on `Shape-Error` before performing the operation.

```lisp
(!add (randn `(3 2)) (randn `(2 4)))
;; will produce...

[cl-waffe] Shaping-Error: Couldn't step forward because of shape-error.

The operation was : <Node: ADDNODE-CPUTENSOR (A[~~] B[~~] -> A[~~])>

Input(s)            : ((3 2) (2 4))
Predicted Output(s) : ((3 2))

Here's a list of reports.

1. Couldn't idenfity ~~: ~~ is determined as 3 
 butgot: 2.
 Excepted ~~ = (3 2), butgot: (2 4)

Also, these reports could be helpful for you (calculated ignoring the first errors.)

2. Couldn't idenfity ~~: ~~ is determined as 2 
 butgot: 4.
 Excepted ~~ = (3 2), butgot: (2 4)
```

### Determine Rules

```
(defnode (ExampleNode (myself)
            :where (A[~~ i j] B[~~ j k] C[~~ k i] -> C[~~ k i])
         ...))
```

Symbols used in subscripts has a two state:

1. Determined (those that can say i=1, j=2!)

2. Undetermined (those that cannot say i=1, j=2)

Before doing `(call (ExampleNode) ...)`, we create a table which stores determined/undetermined symbols and corresponding values.

```
[TABLE]
~~  -> ? // Undetermined before runtime
i  -> ? // Undetermined before runtime
j  -> ? // Undetermined before runtime
k  -> ? // Undetermined before runtime
```

The moment we do `(call (ExampleNode) TensorA TensorB TensorC)`, we will be able to inference the value of `i` `j` `k` from the shape of given TensorA, TensorB, and TensorC.

For Example, Let TensorA be a `2x3x4` Matrix, then the table become:

```
[TABLE]
~~  -> 2
i  -> 3
j  -> 4
k  -> ? 
```

Then continue to do the same thing for TensorB. Let TensorB be a `2x4x9` Matrix, then the table become:

```
[TABLE]
~~ -> 2
i -> 3
j -> 4
k -> 9
```

Last, applying this operation into TensorC, but what if I gave the wrong shape to TensorC? Let TensorC be a `999x999x999` Matrix. (Obviously this is wrong).

```
[TABLE]
~~ -> 2 // ≠999
i -> 3 // ≠999
j -> 4 // ≠999
k -> 9 // ≠999
```

All subscripts in the table do not match with 999, resuting in shape-error.

In that case, we can try again the operation with giving the correct shape to TensorC. Let TensorC be `2x9x3` Matrix.

```
[TABLE]
~~ -> 2 // =2
i -> 3 // = 3
j -> 4 // 
k -> 9 // = 9
```

All subscripts passed! (puts error If there's still undetermined symbol.)

Using the determined table, we can also inference the shape of output tensor. The returned tensor is the shape of `(~~ k i)`, that is, `(2 9 3)`. This operation can be done in a chain of lazy-evaluated nodes.

Now, moving on to another topic, subscripts can be one of them.

```
[TABLE]

a = 1 // Fixnum

b = `(1 2) // List consisted of fixnum

~~ = `(1 2 3) // ~~ is a special symbol which represents batched-input.
```

DSL flattens the list in the subscript. (e.g.: `b=(1 2)` in `A[b]` is the equivalent to `A[1 2]`)

**Note that** ~~ is a reserved word by cl-waffe2 and has a special rule:

1. ~~ is used to express dimensions from 0 to N

2. ~~ can only be used once for one input of subscript.

3. In tables, ~~ is interpreted as one of: `NIL` or `List`

In addition, ~~ has a three behaviour:

1. If ~~ never appears in [Before The Operation] and [After The Operation] parts, the length of ~~ could be Any.

2. If ~~ appears more than once, the length of ~~ and content should be common.

3. If ~~ appears only in [After The Operation], returns error because we can't determine ~~.

In conclusion, I believe introducing Subscript DSL produces two benefits:

1. Rigorous Shape Inspection in all operations with small code, and produce better Shape-Error (Initially I'm inspired in: [nalgebra](https://github.com/dimforge/nalgebra)).

2. JIT Compiler can use a shape of given arguments in advance. (If only CL has a const-generics like Rust, Subscript DSL isn't needed anymore!).

### Initial value of table

In order to give a initial value to tables, you can declare symbols with initial value.

**Using where pharse in :where form**

Add this form to your `:where` form.

```lisp
;; Syntax is that: Symbol-Name = Expression

(defnode (...
    :where (A[i] B[j] -> C[k] where i = 1 j = 2 k = 3)
    ....
```

will produce:

```
[TABLE]
i = 1
j = 2
k = 3
```

Using arguments declared in `constructor`.

```lisp
(defnode (ExampleNode (self i)
             :where (A[~~] -> A[i]))
        ...)
```

Arguments used in constructor, will automatically interpreted as `initial value`. (e.g.: `i` is a initial value.)

```
[TABLE]
~~ = ?
i = i
```

That is, when `ExampleNode` is initialized with `(ExampleNode 3)`, the table become:

```
[TABLE]
~~ = ?
i = 3
```



2. arguments of constructor


### API: create-subscript-p

```(create-subscript-p subscripts &key macroexpand fixed return-body)```

Inputs:

1. macroexpand[Boolean] If t, displays the generated program.

2. fixed[Boolean] If t, ~~ is ignored.

3. return-body[Boolean] If t, the returned is S-exp.

Outputs:

`(values compiled-function To-Refer-Pointer-Idx Broadcastable_List)`

Example: (TODO)

"))

    (with-section "[macro] defnode"
      (insert "

```lisp
(defnode ((abstract-name
		   (self &rest constructor-arguments)
		    &key
		      (where t)
		      (out-scalar-p nil)
                      (save-for-backward nil)
		      (slots nil)
		      (backward nil)
		      (documentation \"\"))
		   &body constructor-body))
```

`defnode` is a macro to define computation nodes in cl-waffe2, which is a subclass of `AbstractNode`.

The class defined is named after `abstract-name`, and they possess the following datum:

1. Generic definition of forward, including `Subscript DSL`, (whch is transimission state of the operation), and `slots` which is shared at forward and backward time.

2. (Optional) Generic definition of backward.

### Inputs

1. `abstract-name` the macro defines a new class named after it.

2. `where`  the place to put Subscript DSL

3. `save-for-backward` corresponding position of input arguments will produce a copy, which is used at backward time.
 
4. `backward` (Optional) Any back-propagation described in define-impl is disabled; instead, the definitions given here are used.

5. `documentation` docstring

6. `out-scalar-p` Set t If the returned tensor is ScalarTensor. This can be dynamically modified via the accessor `(out-scalar-p self)`.

### Effects
 
1. Defines a class named `abstract-name`

2. Defines a function which is used to initialize the node named `abstract-name`

### Useful Tips

In order to simplify parameter initialisation, if the keyword name of the :initarg is the same as the keyword name of the argument, the initialisation code is automatically generated.

```lisp
(defnode (ExampleNode (self arg)
            :slots ((arg :initarg :arg))))

(slot-value (ExampleNode 10) 'arg) ;; => 10
```

### When to define backward?

The backward follows this format:

```lisp
((self dout dx dy ... dn)
 (values dx.grad dy.grad ... dn.grad))
```

`dout` is a previous node's gradient, and `dx dy ... dn` is a variables that used when forward. No guarantee that `dx dy ... dn` isn't being destructed due to in-place operation. If you need them in order to compute gradients, set `:save-for-backward (t t ... t)` at `define-impl` macro.

Find the partial derivative of each variable according to the derivative of the composite function.

The definition of backward must be placed either of defnode or define-impl.
Basically, if the original defnode describes the backward, define-impl's backward is ignored.

```lisp
1.
=================================================================
AddNode (defnode) <- Place Backward
   |
   |-> (AddNode :CPUTensor)  (define-impl)
   |-> (AddNode :LispTensor) (define-impl)
   |-> (AddNode :CUDATensor) (define-impl)
=================================================================

2.
=================================================================
AddNode (defnode) <- Backward=nil
   |
   |-> (AddNode :CPUTensor)  (define-impl) <- place backward
   |-> (AddNode :LispTensor) (define-impl) <- place backward
   |-> (AddNode :CUDATensor) (define-impl) <- place backward
=================================================================
```

Depending on `*using-backend*`, the implementation to use is determined at node-building time. See also: with-devices.

### How to define backward?

```math
g(dout, dx_{in}, dy_{in}, ..., dn_{in}) \\triangleq \\\\
 Move(dx_{out}, {dout} \\times {dx_{grad}}),\\\\
 Move(dx_{out}, {dout} \\times {dy_{grad}}),\\\\
 ...,\\\\
 Move(dx_{out}, {dout} \\times {dn_{grad}})
```

```lisp
:save-for-backward (t t)
:backward ((self dout dx dy)
           (values
               (!mul dout dy)
               (!mul dout dx)))
```

`self` is a place to pass the node class. `dout` is a `AbstractTensor` of previous node's gradient. `dx, dy, ..., dn` are variables used in forward. In the case of the tensor is computed as `In-place`, there's no guarantee that variables aren't destructed. So, to ensure that variables remains as it was, set `:save-for-backward` at corresponding positions if the variable is needed to compute gradient.

According to `the derivative of the composite function`, `:backward` definition should return next node's `dout` following this form:

`(values dx.grad dy.grad ... dn.grad)`

After the computing, cl-waffe2 automatically selects where to store the result, and moves it.

```math
\\begin{equation}
  x_{in}=
  \\begin{cases}
    x_{saveforbackward} & \\text{SaveForBackward is t} \\\\
    \\text{x} & \\text{otherwise}
  \\end{cases}
\\end{equation}
```

```math
\\begin{equation}
  x_{out}=
  \\begin{cases}
    x_{copy} & \\text{If the tensor is a ExistTensor or cause conflicts} \\\\
    \\text{x} & \\text{If the tensor make no conflicts.}
  \\end{cases}
\\end{equation}
```

(`x` is a variable called with `(forward node &rest inputs)` function.)


### Example

```lisp

(defnode (MatMulNode (myself dtype &key transpose-a transpose-b)
	  :where (A[~~ i j] B[~~ j k] C[~~ i k] -> C[~~ i k])
	  :slots ((transpose-a :initarg :transpose-a :type boolean :reader trans-a?)
		  (transpose-b :initarg :transpose-b :type boolean :reader trans-b?))
          :documentation \"gemm\"
	  :backward ((self dout da db do)
		     (declare (ignore do))
		     (values
		      (!matmul dout (!t db))
		      (!matmul (!t da) dout)
		      nil))))

(MatmulNode :float)
;; <Node: MATMULNODE-CPUTENSOR (A[~~ I J] B[~~ J K] C[~~ I K] -> C[~~ I K])>
```

"))

    (with-section "[macro] define-impl"
      (insert "
```lisp
(define-impl ((abstract-name
			&key
			  (device t)
                          (cache-when-compiled t)
			  (reject-p nil))
		       &key
			 save-for-backward
			 forward
			 backward)
```

Gives an implementation to `AbstractNode`.

### Inputs

1. `device` Set here symbol the impl working on. The symbol must be a subclass of `AbstractTensor`. If t, the impl has the highest priority assigned to all implementations.

2. `reject-p[null or predicate]` Set here predicator, If the predicator is t, the implementation refures to be dispatched.

3. `save-for-backward` The corresponding variable which is t will be made a copy when forward. (e.g.: `forward=(x y)` and `save-for-backward=(t nil)`, x is copied, y isn't copied.)

4. `cache-when-compiled[boolean]` If t, `call-with-view` function used in `:forward` will be cached when compiling. Set nil to disable this behaviour.

5. `forward` Place the expanded lisp-code for forward propagation.

6. `backward` Place the definition of backward as the same forward of `defnode` does.

### Tips: reject-p

One of the practical usage of reject-p is to restrict dtypes that implementation can handle.

reject-p takes an function: #'(lambda (&rest inputs) ...) where inputs is `constructor-arguments` in defnode. (e.g.: `(AddNode :float)` -> `inputs=(list :float)`).

`AddNode` for CPUTensor only supports dense matrix.

```lisp
(define-impl (AddNode :device CPUTensor
	     :reject-p (supported-dtypes-are 0 :float :double))
	     :forward ((self x y)
		       `(,@(expand-axpy-form x y)
			     ,x)))
```

The macro `supported-dtypes-are` returns an predicator which returns nil if the first argument is the equivalent to `:float` or `:double`.

### forward/backward

forward/backward is given as:

```lisp
((self &rest arguments)
 body)
```
"))
    
    (with-section "[generic] forward"
      (insert "```(forward node &rest inputs)```
Reading an state of `*using-devies*` and the given nodes, the method `forward` returns a new tensor with applied the forward definition of a given `node` with inputs lazily.

The moment `forward` is called, the computation node is constructed for building forward/backward kernel. Since then, `forward` is `AbstractNode` dedicated operation, not applied into calling `Composite`.

### Example

```lisp
(forward (AddNode :float) (randn `(3 3)) (randn `(3 3)))

{CPUTENSOR[float] :shape (3 3) :named ChainTMP31939 
  :vec-state [maybe-not-computed]
  ((0.109944925 0.42675912  1.9701254)
   (1.5735719   0.7928889   1.1698933)
   (0.08926714  0.0937486   -1.1063566))
  :facet :input
  :requires-grad NIL
  :backward <Node: ADDNODE-CPUTENSOR (A[~~] B[~~] -> A[~~])>}
```
"))

    (with-section "[class] defmodel"
      (insert "~a" (documentation (macro-function 'defmodel) 'function)))
    
    ;;(with-doc 'defmodel 'macro)

    (with-doc 'call 'function)
    
    (with-doc 'with-devices 'macro
      (insert "

```lisp
(with-devices (LispTensor CPUTensor)
   (!add a b))
```"))

    (with-section "[macro] define-and-impl-node"
      (insert "~a" (documentation (macro-function 'define-and-impl-node) 'function)))

    (with-section "[macro] define-composite-function"
      (insert "~a" (documentation (macro-function 'define-composite-function) 'function)))

    (insert "~a" (documentation (macro-function 'define-static-node) 'function))

    (insert "~a" (documentation 'set-save-for-backward 'function))
    (insert "~a" (documentation 'read-save-for-backward 'function))
    
    (insert "~a" (documentation (macro-function 'with-setting-save4bw) 'function))
    (insert "~a" (documentation (macro-function 'with-reading-save4bw) 'function))

    (with-section "[class] Composite"
      (insert
       "
[class] Composite

~a"
       (documentation (find-class 'Composite) 't)))

    (with-section "[class] AbstractNode"
      (insert
       "
[class] AbstractNode

~a"
       (documentation (find-class 'AbstractNode) 't)))

    (with-doc 'with-instant-kernel 'macro)))

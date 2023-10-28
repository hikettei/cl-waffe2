
(in-package :cl-waffe2.docs)

(with-page *nodes* "Formulating Computation Nodes"
  (insert "The package `:cl-waffe2/vm.nodes` provides a features on `AbstractNode` and `Composite`, which is a fundamental data structure to represent computation node. `AbstractNode` is the smallest unit of the operation in the network, and `Composite` is a class which bundles several `AbstractNodes` (`Composite=nn.Module or Model` in other frameworks).

The role of node and model is completely different. To perform operations with `AbstractNode` we have to step a two steps: `General Definition` and `Device Specific Implementations`. `AbstractNode` is defined by the macro `defnode` with its specifications but forward implementation. The macro `define-impl` or `define-impl-op` will provide device-specific implementations for each AbstractTensor. The differences between `define-impl` and `define-impl-op` is that: The :forward definition is given by a macro or a function respectively. With `define-impl` macro and the `call-with-view` function, you can create a operation which optimizes, fuses, and collapses  the order of iteration, schedules multi-threading, and computes view offsets in advance with a simple form.

On the other hand, since `Composite` is user to bundle several nodes, it is not only used to juse represent Neural Network Model (e.g.: `LinearLayer` `Conv2D`...) but compiled into `function` or `AbstractNode` with the `defmodel-as` macro by tracing its computation node declared in the `:call->` method. However, to do this, you have to declare these information in advance: `The rank/shape of tensors, the number of arguments, and which operations are In-place?`. This is also true for `AbstractNode` and cl-waffe2 introduced an small DSL to represent this, `Subscript DSL (:where ...)`.

In short word, Subscript DSL is used to:

- For AbstractNode, declares the transmission states of the operation (MUST)

- For Composite, in order to trace the network, it declares the transmission state of operation (Optional)

Accordingly, this document is divided to three sections.

- The specification of Subscript DSL

- AbstractNode

- Composite

And an overview of APIs is here:

```lisp
[AbstractNode] The fundamental unit of forward/backward propagations.
 defnode - Declares a general definition of AbstractNode
      L define-impl     Implements a AbstractNode. Its forward definition is given as a macro (to inline/call-with-view), later (compile nil body) is called, and cached when :compile-when-cache=t.
      L define-impl-op  Implements as a lambda function.

define-op = defnode + define-impl-op

[Composite] Bundles several AbstractNodes, defined by defmodel macro.
  defmodel - Defines a new Composite
     L defmodel-as Redefining the existing Composite as a function or AbstractNode to reduce compiling time, to use cl-waffe2 as a define-by-run library.
```

cl-waffe2 VM sorts and compiles the network of `AbstractNode` into a `cl-waffe2 IR (Extended Wengert List)` and operations are performed. And, `AbstractNode` is used to represent an blueprint of lambda functions. Both of AbstractNode and Composite are the CLOS class.
")

  (macrolet ((with-doc (name type &body body)
	       `(progn
		  (placedoc ,name ,type)
		  ,@body)))

    (with-section "Representing shapes before and after the operation."
      (insert "
When defining an operation in cl-waffe2 with a `defnode` macro, the shape of the matrix used in the operation must also be defined in the `:where` keyword.

This is a Shaping API, and responsible for shape inspection of all operations, and tracing the network.

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
 but got: 2.
 Expected ~~ = (3 2), but got: (2 4)

Also, these reports could be helpful for you (calculated ignoring the first errors.)

2. Couldn't idenfity ~~: ~~ is determined as 2 
 but got: 4.
 Expected ~~ = (3 2), but got: (2 4)
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

    (with-section "AbstractNode")

    (insert (documentation (find-class 'AbstractNode) 't))

    (with-doc 'defnode 'macro)
    (with-doc 'define-impl 'macro)
    
    (with-doc 'define-impl-op 'macro)
    (with-doc 'define-op 'macro)
    
    (with-doc 'set-save-for-backward 'function)
    (with-doc 'read-save-for-backward 'function)

    (with-doc 'with-reading-save4bw 'macro)
    (with-doc 'with-setting-save4bw 'macro)
    
    (with-section "Composite")

    (insert (documentation (find-class 'Composite) 't))

    (with-doc 'defmodel    'macro)
    (with-doc 'defmodel-as 'macro)

    (with-section "Symbolic Diff")

    (insert "~a" (documentation '*enable-symbolic-path* 'variable))
    (with-doc 'define-symbolic-path 'macro)
    (with-doc 'define-bypass 'macro)

    (with-section "Events for Embedding JIT-Generated Code in runtime")
    
    (insert "~a"
	    (documentation #'on-finalizing-compiling 't))
    
    ))

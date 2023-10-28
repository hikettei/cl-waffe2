
# Formulating Computation Nodes
The package `:cl-waffe2/vm.nodes` provides a features on `AbstractNode` and `Composite`, which is a fundamental data structure to represent computation node. `AbstractNode` is the smallest unit of the operation in the network, and `Composite` is a class which bundles several `AbstractNodes` (`Composite=nn.Module or Model` in other frameworks).

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

## Representing shapes before and after the operation.

When defining an operation in cl-waffe2 with a `defnode` macro, the shape of the matrix used in the operation must also be defined in the `:where` keyword.

This is a Shaping API, and responsible for shape inspection of all operations, and tracing the network.

## Introducing Subscript DSL

I assume you have already seen `defnode` macro. This macro takes a strange syntax language after :where keyword.

```lisp
(defnode (TransposeNode (myself)
            :where (A[~ i j] -> A[~ j i])
         ...))

(defnode (ScalarAdd (myself)
            :where (A[~] Scal[scal] -> A[~] where scal = 1)
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

// SubScripts can be one of: [A], [A B] [~ i j] etc...
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

The operation was : <Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>

Input(s)            : ((3 2) (2 4))
Predicted Output(s) : ((3 2))

Here's a list of reports.

1. Couldn't idenfity ~: ~ is determined as 3 
 but got: 2.
 Expected ~ = (3 2), but got: (2 4)

Also, these reports could be helpful for you (calculated ignoring the first errors.)

2. Couldn't idenfity ~: ~ is determined as 2 
 but got: 4.
 Expected ~ = (3 2), but got: (2 4)
```

### Determine Rules

```
(defnode (ExampleNode (myself)
            :where (A[~ i j] B[~ j k] C[~ k i] -> C[~ k i])
         ...))
```

Symbols used in subscripts has a two state:

1. Determined (those that can say i=1, j=2!)

2. Undetermined (those that cannot say i=1, j=2)

Before doing `(call (ExampleNode) ...)`, we create a table which stores determined/undetermined symbols and corresponding values.

```
[TABLE]
~  -> ? // Undetermined before runtime
i  -> ? // Undetermined before runtime
j  -> ? // Undetermined before runtime
k  -> ? // Undetermined before runtime
```

The moment we do `(call (ExampleNode) TensorA TensorB TensorC)`, we will be able to inference the value of `i` `j` `k` from the shape of given TensorA, TensorB, and TensorC.

For Example, Let TensorA be a `2x3x4` Matrix, then the table become:

```
[TABLE]
~  -> 2
i  -> 3
j  -> 4
k  -> ? 
```

Then continue to do the same thing for TensorB. Let TensorB be a `2x4x9` Matrix, then the table become:

```
[TABLE]
~ -> 2
i -> 3
j -> 4
k -> 9
```

Last, applying this operation into TensorC, but what if I gave the wrong shape to TensorC? Let TensorC be a `999x999x999` Matrix. (Obviously this is wrong).

```
[TABLE]
~ -> 2 // ≠999
i -> 3 // ≠999
j -> 4 // ≠999
k -> 9 // ≠999
```

All subscripts in the table do not match with 999, resuting in shape-error.

In that case, we can try again the operation with giving the correct shape to TensorC. Let TensorC be `2x9x3` Matrix.

```
[TABLE]
~ -> 2 // =2
i -> 3 // = 3
j -> 4 // 
k -> 9 // = 9
```

All subscripts passed! (puts error If there's still undetermined symbol.)

Using the determined table, we can also inference the shape of output tensor. The returned tensor is the shape of `(~ k i)`, that is, `(2 9 3)`. This operation can be done in a chain of lazy-evaluated nodes.

Now, moving on to another topic, subscripts can be one of them.

```
[TABLE]

a = 1 // Fixnum

b = `(1 2) // List consisted of fixnum

~ = `(1 2 3) // ~ is a special symbol which represents batched-input.
```

DSL flattens the list in the subscript. (e.g.: `b=(1 2)` in `A[b]` is the equivalent to `A[1 2]`)

**Note that** ~ is a reserved word by cl-waffe2 and has a special rule:

1. ~ is used to express dimensions from 0 to N

2. ~ can only be used once for one input of subscript.

3. In tables, ~ is interpreted as one of: `NIL` or `List`

In addition, ~ has a three behaviour:

1. If ~ never appears in [Before The Operation] and [After The Operation] parts, the length of ~ could be Any.

2. If ~ appears more than once, the length of ~ and content should be common.

3. If ~ appears only in [After The Operation], returns error because we can't determine ~.

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
             :where (A[~] -> A[i]))
        ...)
```

Arguments used in constructor, will automatically interpreted as `initial value`. (e.g.: `i` is a initial value.)

```
[TABLE]
~ = ?
i = i
```

That is, when `ExampleNode` is initialized with `(ExampleNode 3)`, the table become:

```
[TABLE]
~ = ?
i = 3
```



2. arguments of constructor


### API: create-subscript-p

```(create-subscript-p subscripts &key macroexpand fixed return-body)```

Inputs:

1. macroexpand[Boolean] If t, displays the generated program.

2. fixed[Boolean] If t, ~ is ignored.

3. return-body[Boolean] If t, the returned is S-exp.

Outputs:

`(values compiled-function To-Refer-Pointer-Idx Broadcastable_List)`

Example: (TODO)


## AbstractNode


## [class] AbstractNode

AbstractNode is a CLOS class to represent operations.

Can be created by a function `(AbstractName ...)` declared by the defnode macro.

In order to step the computation: `(forward node arg1 arg2 ...)` (using a `call` instead of `forward` is ok)

And backward: `(backward node prev-gradient arg1 arg2 ...)`


## [macro] defnode

```lisp
(defnode (abstract-name
		   (self &rest constructor-arguments)
		    &key
		      (where t)
		      (out-scalar-p nil)
		      (slots nil)
		      (save-for-backward nil)
		      (backward nil)
		      (extends nil)
		      (documentation ""))
		   &body constructor-body)
```

Declares a new `AbstractNode`.

### Effects
   - defines a class (subclass of `AbstractNode`) named `abstract-name`
   - defines a fucntion which initializes the defined node.

### Inputs

- `abstract-name`[symbol] indicates the name of class, and constructor.
- extends[list] set a list of symbols, the class is defined with extending them.

- `(self &rest constructor-arguments)` declares the arguments of the constructor function, which `cosntructor-body` uses. 

- slots[list] Describe the slots which node has as if defclass. Tips: In order to make it shorter to create a constructor, if initargs (i.e.: `:initarg :XXX`) is the same as the keyword name of the argument, the initform is replaced with the argument.

- where[SubscriptDSL] Put here the Subscript DSL (MUST)

- out-scalar-p [Boolean] Set t if the node returns a ScalarTensor.

- backward [list] This form is optional. The backward receives arguments like: `(dout var1 var2...)` and return tensors which is lazy-evaluated. (See examples). You can set this form as nil, but in that case each `define-impl` and `define-impl-op` must have a backward slot.

- documentation [String]

### Example

```lisp
;; Tips
(defnode (ExampleNode (self arg)
            :slots ((arg :initarg :arg))))

(slot-value (ExampleNode 10) 'arg) ;; => 10

(defnode (MatMulNode-Revisit (myself dtype &key transpose-a transpose-b)
	  :where (A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :slots ((transpose-a :initarg :transpose-a :type boolean :reader trans-a?)
		  (transpose-b :initarg :transpose-b :type boolean :reader trans-b?))
	  :backward ((self dout da db do)
                     ;; dout=previous gradient, :save-for-backward is set to (t t nil).
                     ;; so da/db is a copy of variable.
		     (declare (ignore do))
                     ;; Set nil to the direction gradients aren't produced.
		     (values
		      (!matmul dout (!t db))
		      (!matmul (!t da) dout)
		      nil))
	  :documentation "
C <- GEMM(1.0 A B 0.0 C)
"))

```

You can invoke the forward/backward by using the method forward/backward. `(forward node arg1 arg2...)` `(backward node dout1 arg1 arg2...)`.

## [macro] define-impl

```lisp
(define-impl (abstract-name &key (device t) (extends nil) (cache-when-compiled t) (reject-p nil) (cache-id nil))
        &key (save-for-backward nil) (forward nil) (backward nil))
```

Defines a one of implementation of `abstract-name` which is defined by `defnode` macro. The implementation is given as the same manner of defmacro. Returned S-expression is later compiled by the `(compile nil body)` function and cached as long as cache-when-compiled is set to T. Compiled functions are dispatched depending on `RANK` `DTYPE` `STRIDE` and `SHAPE`, if you want to add another factors this, specify this at :cache-id.


### Effects

Defines a CLOS class named `abstract-name-device` extends `abstract-name`

### Inputs

`device`[symbol or t] Set the name of AbstractTensor which the impl supports for. Set t to anything.

`extends`[nil or list] In addition to extend `abstract-name`, the defined implementation will extends the given classses.

`cache-when-compiled`[boolean] Set T to cache the forward definiton depending on dtypes, ranks, devices of arguments. You can set this to NIL but in terms of performance it is not recommended (runtime-compiling overhead is unignorable!) Instead, in that case, using `define-impl-op` would be nice.

`save-for-backward`[list of boolean] For backward computation, the corresponding position of received variables will be produce a copy. You can check how it works with `(disassemble-waffe2-ir toplevel)` function and SV4BW(...) is exactly this. In backward, `((self dout x y) ...)` will receive the copy.

`forward`[body] Follows this format: `((self arg1 arg2 ...) <<macro-body>>)` and the form must return S-expression later compiled by `(compile nil ...)

`backward`[body] Follows this format: `((self prev-gradient arg1 arg2 ...) (values arg1.grad arg2.grad))` Note that the form is given by a function, and computation nodes are continuous. Not a macro.

`reject-p`[nil or function] Set a lambda function returning nil or T. The function is called with arguments: `(function constructor-args1 constructor-args2 ...)`. In the case the function returned T, the method dispatching is ignored. You can use this method to ignore a certain dtype as a :forward arguments for example.

`cache-id[nil or function]` Adds an additional keys of searching LUT. this form should be given as: `#'(lambda (&rest self inputs) (list keys...))` where inputs are the arguments called with forward. For example: `#'(lambda (self &rest inputs) (map 'list #'order inputs))` if the orders matter.

## [macro] define-impl-op

Gives an implementation of `abstract-name` as a function form.

```lisp
(define-impl-op ((abstract-name &key (device t) (extends nil) (reject-p nil)) &key forward backward))
```

In order to place ranked matrix operations here, you MUST use `do-compiled-loop` macro instead of writing iterations manually.

## [macro] define-op

`define-op` = `defnode` + `define-impl-op`

Defines a differentiable AbstractNode which its definition is given by a function.

```lisp
(define-op (name (self &rest constructor-args) where slots out-scalar-p save-for-backward-names forward backward documentation extends) &body body)
```

### Effects

This macro defines:

1. two `AbstractNodes` named `name` and `name-backward` (if backward is given)

### Example

```lisp
(define-op (TestAdd-Scalar (self)
	    :where (A[scal] B[scal] -> A[scal] where scal = 1)
            :out-scalar-p t
	    :forward ((self a b)
		      (make-tensor
		       (+ (tensor-vec a)
			  (tensor-vec b))))
	    :backward ((self dy)
		       (values dy dy))))
```

## [function] set-save-for-backward

```lisp
(set-save-for-backward self name tensor)
```

The function `set-save-for-backward` saves the given `tensor` to the `name` slot of self for a future call of backward.

This function is dedicated to the macro `define-static-node`, so it should be placed at the forward/backward definition of the macro, otherwise, the wrong function is binded which returns simple-error. In addition, The place to save the tensor, should be also declared in `:save-for-backward-names` in the `define-static-node` macro.

Note that this function is ignored in specific conditions: `*no-grad*` is t or `set-save-for-backward` in the forward definition in the forward definition. (i.e.: the place which is never called.)

See also: `read-save-for-backward` `with-setting-sv4bw` `with-reading-sv4bw` `define-static-node`

## [function] read-save-for-backward

```lisp
(read-save-for-backward self name)
```

Reading the slot of `name` in `self`, the function `read-save-for-backward` returns a saved tensor by `set-save-for-backward`.

For the same reason of `set-save-for-backward`, this function should be placed at right place.

## [macro] with-reading-save4bw

```lisp
(with-reading-save4bw ((&rest input-forms) &body body))

input-form = (variable-place save-for-backward-name)
```

Reading the save-for-backward of currently working node, the macro binds each `variable-place` the stored tensor.

## [macro] with-setting-save4bw

```lisp
(with-setting-save4bw ((&rest input-forms) &body body))
input-form = (save-place tensor)
```

Saves the given tensors to save-place, in the currently working node.

## Composite


## [class] Composite

Its `call` bundles several AbstractNode. It is not only used to represent a neural network but also convert nodes into functions or abstractnodes. You can forward composites with `(call composite arg1 arg2...)`. For the most case, composites are defined by the defmodel macro.

### [generic] on-print-object

```
(on-print-object model stream)
```

This generic function is used to customize how the model is printed on the display.

```
<Composite: NAME{...}(
    [...] <- The content of here is depends on on-print-object
    [PARAMTETERS]
)
```

## [macro] defmodel
```
(defmodel ((name
	     (self-name &rest constructor-arguments)
		      &key
		       (slots nil)
		       (initargs)
                       (where nil)
		       (on-call-> nil)
		       (documentation ""))
		    &body constructor-body)
```

Defines a composite named `name`, and constructor function which also named `name` and receives `constructor-arguments` as arguments. The main process of its forward process is described in the `on-call->` slots.

### Inputs

`name[Symbol]` the macro defines an class and constructor function named after it.

`(self-name &rest constructor-arguments)` An initializer form of `constructor function`.

`slots ((slot-option1) (slot-option2) ...)` Parameters of the inherited Composite class. It has the same syntax as defclass slots`

`initargs (:accessor-name1 accessor-init-form1 :accessor-name2 accessor-init-form2 ...)` Unlike structures, CLOS classes are somewhat more cumbersome to initialise parameters. To make this process simple, put here initializer forms in advance likewise we do `(make-instance class-name ...)`.

`documentation[String]`

`on-call-> [One of: nil symbol-name function list]` The main proces of its forward process, later called with `(call model ...)` method. This method must be continuous from the given arguments.

`where[Subscript DSL] (Optional)` If you're planning to use `defmodel-as` macro, this form is needed.

### Example

```lisp
(defmodel (ExampleLayer (self features)
               ;; Options/Utils Here,
               :slots    ((param :initarg :param))
               :initargs (:param (make-tensor `(,features) :requires-grad t))
               :documentation "ExampleLayer is a ...")

    ;; After make-instance is called, the form below is called.
    ;; make-instance -> make-instance :after -> this form.

    (print self)     ;; <- Initialized ExampleLayer
    (print features) ;; <- constructor-arguments are also used here.
    (print "ExampleLayer is created!"))

;; The model you created, works like:
(let ((layer (ExampleLayer 10)))
    (call layer ...))
```

```lisp
(defmodel (Softmax-Model (self)
	   :where (X[~] -> [~])
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (let* ((x1 (!sub x (!mean x  :axis 1 :keepdims t)))
	                      (z  (!sum   (!exp x1) :axis 1 :keepdims t)))
                           (!div (!exp x1) z)))))

;; Keep Using Lazily...
(proceed (call (Softmax-Model) (randn `(10 10)))
{CPUTENSOR[float] :shape (10 10) :named ChainTMP33497 
  :vec-state [computed]
  ((0.04800622   0.118814774  0.050377533  ~ 0.053051848  0.050124187  0.25575548)                    
   (0.15909052   0.11368358   0.12642372   ~ 0.114795394  0.033397682  0.07605342)   
                 ...
   (0.035624444  0.24828684   0.109363265  ~ 0.020787988  0.027314318  0.04515641)
   (0.030307569  0.24117047   0.03900468   ~ 0.014522874  0.036584295  0.0971196))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}

(defmodel-as (Softmax-Model) :asif :function :named softmax-static)

;; No compiling overhead
(softmax-static (randn `(10 10)))

{CPUTENSOR[float] :shape (10 10) :named ChainTMP33788 
  ((0.16722792   0.018530384  0.014159603  ~ 0.035353966  0.06128503   0.13559735)                    
   (0.14498742   0.11881006   0.0692616    ~ 0.03911829   0.10358454   0.02131605)   
                 ...
   (0.055657785  0.44042623   0.030706322  ~ 0.11048273   0.0097645    0.11959953)
   (0.059088983  0.11067564   0.120767005  ~ 0.15042976   0.06570089   0.20548664))
  :facet :input
  :requires-grad NIL
  :backward NIL}
```

### Dispatching on-call-> method

- `on-call-> is nil` In that case, users must define the call definiton manually like `(defmethod call ((model YourComposite) arg1 arg2) ...)`.

- `on-call-> is symbol` In that case, the composite invokes the method named `symbol` when call is invoked.

```lisp
;; Set :on-call-> call-example-layer


(defmethod call-example-layer ((model ExampleLayer) x y)
    (print "call-example-layer is used!"))
```

```lisp
(call (ExampleLayer 10) tensor) ;; call-example-layer is used!
```

### `on-call->` is a list

Directly defines a `call` method. Arguments must be: `(self arg1 arg2...)`
```lisp
...
    :on-call-> ((self x)
                (!sin x))
```

## [macro] defmodel-as

```lisp
(defmodel-as target-model &key (where nil) (asif :function) (named nil) (differentiable nil))
```

Redefines a Composite as a new function or AbstractNode specified in the `:asif` keyword. Further functions or `Differentiable AbstractNode` can be defined based on existing Composites (also called as `model` and defined by `defmodel` macro) which bundles several `AbstractNodes`, as long as `:where` form is fulfilled.

### Example

```lisp
(defmodel-as (SoftmaxNode) :named static-softmax :asif :function :where (A[~] -> A[~]))
```

### Inputs

`target-model[Composite]` a form to initialize the composite. ~~This from is executed before running the code, and accordingly static.~~

`where[Subscript DSL or null]` If the model has no `:where` declaration, this macro uses this `:where` form instead. Therefore, as long as `defmodel` provides `:where` declaration, this form should be OK if set as nil.

`named[symbol]` this macro will define a new function after `named`. If set to `nil`, the macro return a lambda function instead of defining it. If you're trying to define a new `AbstractNode`, this option should be fulfilled.

`:asif[keyword]` indicates which form the `target-model` is to be redefined, and could be one of:

```
─────────────────────────────────────────────────────────────────────────────────────
  asif    |   description
─────────────────────────────────────────────────────────────────────────────────────
:function | Defines a function to be executed immediately that does not create a node.
─────────────────────────────────────────────────────────────────────────────────────
:node     | Defines a AbstractNode which needs to be compiled later
─────────────────────────────────────────────────────────────────────────────────────
```

### Effects

If `named` is not `NIL`, this macro defines a new function or AbstractNode after `named`.

### Notes

Depending on the `device` and `dtype` used of arguments, several methods are compiled and dispatched.

## Symbolic Diff

## [parameter] `*enable-symbolic-path*`

Indicates function calls replaced with `define-symbolic-path` and `define-bypass` effects on the result.

Set T to enable symbolic diff. In default: `T`.

## [macro] define-symbolic-path

```lisp
(define-symbolci-path (subject &rest clause) (&key (device t) (env (gensym))) (&rest form-binds) &body replacement)
```

Defines a compiler-macro so-called **Symbolic Differentiation** which fuses several nodes into one, or replaces with another nodes. Sometimes, can combine cl-waffe2 functions to compose bad a computation node in tern of speed and safety; nodes (e.g: `(log (1+ x))`, `(log (exp x))`) should be represented as `(log1p x)` or `x` in the first place for reverse mode autodiff, and some nodes like `(!div X X)` should be deleted before compiling. This macro, however, enables that detecting such combinations and replacing them with another node before compiling.

First, describe `subject` the function name to be replaced (e.g.:`!log` `!sum` `!sin` etc...). And then, each `caluses` receive an argument of corresponding position, and determine if the form can be replaced or transformed. Plus, currently using devices can be included to the condition: Only after a car of `*using-backend*` is a subtype of `device`, symbolic path is replaced. At the last, each result of `clauses` will be binded to `form-binds`, and return the improved code at the `replacement` in a manner of `defmacro.` If needed, `&environment` is binded to the `env`.

### Inputs

- `clauses[form]` `((var) body)`

### Effects

- defines a compiler-macro named after `subject`.

### Example

Considering composing `(!!log (!!+ x 1))`

```lisp
(defun !!log (x)
  (print "LOG")
  (log x))

(defun !!+   (a b)
  (print "+")
  (+ a b))

(defun !!log1p (x)
  (print "LOG1P")
  (log (+ 1 x)))
```

```lisp
(define-symbolic-path (!!log
		       ((x)
			(trivia:match x
			  ((or (list '!!+ 1 var)
			       (list '!!+ var 1))
			   var))))			
    (:device cl-waffe2/backends.cpu:CPUTensor)
    (x)
  `(!!log1p ,x))
```

```lisp
(defun test-form (x)
  (!!log (!!+ 1 (!!log x))))
```

```lisp
(test-form 2)
;; LOG
;; LOG1P
;; 0.52658904
(test-form 2)
(setf *enable-symbolic-path* NIL) ;; Disable this feature
;; LOG
;; +
;; LOG
;; 0.52658904
```

Since the macro defines a compile-macro, this optimizing feature can be added one per one function. For example, If the purpose is to replace the standard implementation of `cl-waffe2/nn:!relu` with another fused and device-specific implementation, use the `define-bypass` macro.

## [macro] define-bypass

```lisp
(define-bypass device name replacement)
```

Defines a compiler-macro called **bypass**, which replaces an existing function call with another one. If you want to fuse nodes created by functions which creates computation node (e.g.: !relu !softmax !gelu), declare an alternative route with this function, and they can be replaced with like: ReLUNode, SoftmaxNode, GeLUNode.

The replacing is done when one of `*using-backend*` is the equivalent to `name[symbol]`, the funcall of `name[symbol]` will be replaced with `replacement[symbol]`. Note that before and after the replacement, they both should take the same arguments, same keywords. Unlike `define-symbolic-path`, there is no restriction of numbers that can be registered as a bypass to the single function; A single `!relu` can be replaced with: `!relu-cpu-fuse`, `!relu-cuda-fuse` for example.

### Example

```lisp
(defun !!relu (x)
  (print "RELU")
  x)

(defun !!relu-fuse (x)
  (print "RELU_FUSE")
  x)

(defun op (x)
  (!!relu x))

(define-bypass cl-waffe2/backends.cpu:CPUTensor !!relu !!relu-fuse)

(op 3)
;; RELU_FUSE
;; 3

(setf *enable-symbolic-path* nil)

(op 3)
;; RELU
;; 1
```


## Events for Embedding JIT-Generated Code in runtime

## [generic] on-finalizing-compiling

```lisp
(on-finalizing-compiling device-name iseq-fw iseq-bw)
;; => Return: (values iseq-fw-new iseq-bw-new)
```

This method enables external backends to rewrite the generated cl-waffe2 IR. In order to implement features like `JIT Compiling to C/CUDA`, or `Fusion OP`, external backends may need to rewrite IR, this method is intended to be used so. Concretely, at the point where compilation is finished, this method is called in the order of the device priority symbols being dispatched by the `device-name` argument. InstructionSeq generated by vanila cl-waffe2 compiler, is later replaced by `(values iseq-fw-new iseq-bw-new)` respectively. This operation is repeated for all devices. Ignored if the method not found.

If you want to add an additional optimizing, the method can be used like:

```lisp
(defmethod on-finalizing-compiling ((device-name (eql 'MyTensor)) iseq-fw iseq-bw)
    ;; (Optimizing Process...)
    (values iseq-fw iseq-bw))
```

### Inputs

`device-name[symbol]` indicates the name of device being used to dispatch the method.

`iseq-fw iseq-bw[list]` InstructionSeq. an list of `WfInstruction(See the section of cl-waffe2/vm)`.

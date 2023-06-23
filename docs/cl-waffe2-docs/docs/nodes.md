
# Formulate Neural Networks
The package `:cl-waffe2/vm.nodes` provides a fundamental system for building neural networks.

This package can be divided into three main parts.

1. Shaping APIs
2. defnode  (Differentiable Operations)
3. defmodel (Operations consisted of defnode)

Note that there's a clear distinction between node and model.

```lisp
defnode  => called with `forward` 
defmodel => called with `call`
```

Also, defnode is a fundamental unit of operation, while defmodel is consisted of a set of nodes.

## Shaping APIs


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

1. local variables declared in `defnode` form or `where` pharse.

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
 butgot: 2.
 Excepted ~ = (3 2), butgot: (2 4)

Also, these reports could be helpful for you (calculated ignoring the first errors.)

2. Couldn't idenfity ~: ~ is determined as 2 
 butgot: 4.
 Excepted ~ = (3 2), butgot: (2 4)
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

**Note that** ~ is a reserved word and has a special rule:

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

### Where Pharse

With where pharse, you can put local variables like:

```lisp
;; Syntax is that: Symbol-Name = Expression

A[i] B[j] -> C[k] where i = (1+ (random 1)) j = (1+ (random 1)) k = (1+ (random 1))
```

### API: create-subscript-p

```(create-subscript-p subscripts &key macroexpand fixed return-body)```

Inputs:

1. macroexpand[Boolean] If t, displays the generated program.

2. fixed[Boolean] If t, ~ is ignored.

3. return-body[Boolean] If t, the returned is S-exp.

Outputs:

`(values compiled-function To-Refer-Pointer-Idx Broadcastable_List)`

Example: (TODO)


## defnode

```lisp
(defnode ((abstract-name
		   (self &rest constructor-arguments)
		    &key
		      (where t)
		      (out-scalar-p nil)
		      (slots nil)
		      (backward nil)
		      (documentation ""))
		   &body constructor-body))
```

defnode is a macro which is used to define a subclass of `AbstractNode`.

The defined class is named after `abstract-name`, which has:

1. Subscript DSL

2. Slots that are shared at forward/backward time.

3. Generic definition of backward

### Inputs

1. `abstract-name` the class is named after it

2. `where`  the place to put Subscript DSL

3. `backward` the general definition of backward (Optional). Place S-expression here If you wanna ignore define-impl's backward, otherwise define-impl's one is used.

4. `documentation` docstring

5. `out-scalar-p` Set t If the returned tensor is ScalarTensor. This can be dynamically modified via the accessor `(out-scalar-p self)`.


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

### How and When to define backward?

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
## define-impl

```lisp
(define-impl ((abstract-name
			&key
			  (device t)
			  (reject-p nil))
		       &key
			 save-for-backward
			 forward
			 backward)
```

Defines a implementation of AbstractNode of `device`.

### Inputs

1. `device` Set here symbol the impl working on. The symbol must be a subclass of `AbstractTensor`. If t, the impl has the highest priority assigned to all implementations.

2. `reject-p[null or predicate]` Set here predicator, If the predicator is t, the implementation refures to be dispatched.

3. `save-for-backward` The corresponding variable which is t will be made a copy when forward. (e.g.: `forward=(x y)` and `save-for-backward=(t nil)`, x is copied, y isn't copied.)

4. `forward` Place the expanded lisp-code for forward propagation.

5. `backward` Place the definition of backward as the same forward of `defnode` does.

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

## forward
```(forward node &rest inputs)```
Step forward of the given `node`, node is a subclass of `AbstractNode`.

Note that `forward` can't handle with `Composite`.
## defmodel

```
(defmodel ((name
	     (self-name &rest constructor-arguments)
		      &key
		       (slots nil)
		       (initargs)
		       (on-call-> nil)
		       (documentation ""))
		    &body constructor-body)
```

defmodel is a macro used to describe the model of neural network with `Composite` class.

### Effects

   1. defines a class named **name**

   2. defines a function named **name** with the constructor-arguments and constructor-body.


### Inputs

  1. name[Symbol]  All models, and constructors for the model, are named after it.
  2. (self-name &rest constructor-arguments)
    The constructor function is defined as:
    (defun ,name (self-name ,@constructor-arguments)
       ...)

  3. slots ((slot-option1) (slot-option2) ...)
    Parameters of the inherited Composite class. It has the same syntax as defclass slots

  4. initargs (:accessor-name1 accessor-init-form1 :accessor-name2 accessor-init-form2 ...
    Unlike CL's structure, classes are tend to rebundant when writing the process of initializing slots. To make this simple, this argument was introduced. It works like a structure's constructor!

  5. documentation[String]

  6. `on-call->` [One of: nil symbol-name function list]
     on-call-> is used to control the behaviour of *call* function.

  7. `on-print-object` [null or body]

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

### Describe Forward Propagation

The option `on-call->` can control the behaviour of *call* function.

`on-call->` could be one of these case:

First case,  `on-call->` is nil:

  cl-waffe2 calls the **call** function when doing forward propagation of the model.

Second case, `on-call->` is symbol-name:

   cl-waffe2 calls the specified function at on-call-> parameter, when doing forward propagation of the model.

   symbol-name could be also one of: method's name function's name.

   For example, set `:on-call-> = call-example-layer` which defined as:

```lisp
   (defmethod call-example-layer ((model ExampleLayer) x y)
       (print "call-example-layer is used!")
       ...)
```


```lisp
   (call (ExampleLayer 10) tensor) ;; call-example-layer is used!
```

   (Complex model assignments like ConvND, for example, can be achieved by assigning generic function names to symbols.)

[Third case] `on-call->` is function (i.e.: lambda):

   cl-waffe2 calls the given lambda function as a forward propagation.

[Fourth case] `on-call->` is a list:

   The List, should be this format.

   `((arguments) body)`

   This argument is expanded into `#'(lambda ,@on-call->)` and works as well as 3.
## call
All models in cl-waffe2, should implement this generic function. This generic function returns the computation node of the forward propagation of the model.

The generic function call is also used to step forward of AbstractNode, that is, works as if forward.

`[generic-function]` (call model &rest inputs)
## with-devices
The macro with-devices declares the node's priority for the function *forward* to be used.

Input:
   - backend-priority
     An list of device's name (e.g.: CPUTensor, LispTensor...)
     Devices on the left have higher priority.

Example:

Let ATensor and BTensor be compatible (i.e.: pointers are the same type), and subclass of AbstractNode, and all the operations they have are as follows:

1. ATensor has !add.
2. BTensor has !mul.

This code works:

(setq a (make-tensor `(10 10))) ;; The tensor a is ATensor.

;; (Priority1=ATensor Priority2=BTensor)
(with-devices (ATensor BTensor)
   (!add a (!mul a a)))

ATensor doesn't have any implementation of !mul, but it does work. This is because cl-waffe2's compatible backend system.

cl-waffe2's backend dispatching rule is following:

If the priority 1 backend does not have an implementation of the specified operation, check if the priority 2 backend does, if it still does not have it, 3, 4... and so on.

The order of priority would be `(,@backend-priority ScalarTensor t). (t is a special name, and it implys the implement works for all the backends.)

### Example

```lisp
(with-devices (LispTensor CPUTensor)
   (!add a b))
```
## Composite

[class] Composite

Composite is a fundamental datatype for all neural network models. The name composite is so named because it is used to bundle computation nodes constructed by defnode.

In cl-waffe2, All models should be a subtype of this class, and shall return a forward propagation computation node using the **call** function.

In order to define your model with Composite, two methods are available.

### Extend Composite Class (Slightly Complicated)

First, define your class with extending Composite Class.

```lisp
(defclass LinearModel (Composite)
   ((weight ...) ; <- set parameters here.
    (bias   ...))
```

Second, define forwarrd step with overriding call method.

```lisp
(defmethod call ((model LinearModel) &rest inputs)
     ... )
```

It should work like:

```(call (make-instance 'LinearModel in-features out-features) args1 ...) ```

### Using defmodel macro

The defmodel macro simplifies the above redundant notation and also solves the problem that call can only use &rest as an argument. Therefore, I'm depcrecated with the method above, instead, use defmacro. For detailed usage, see the documentation of defmacro.

## AbstractNode

[class] AbstractNode

The class AbstractNode is a fundamental object of describing computation nodes in cl-waffe.

AbstractNode must possess following:

   1. Transimission State

   2. Slots (for passing forward/backward)

   3. Variables (for building computation nodes)

## with-instant-kernel
Creates an instant-kernel following tensor.

This macro is used to embed condition-free Lisp code either in the process of creating a node or after it has been compiled.

Use case:

### Embedding Lisp Code for building-time.

```lisp
(setq a (randn `(10 10)))
(with-instant-kernel a
    (print a)) ;; -> (print a) is evaluated
```

### Embedding Lisp Code for compile-time.

```lisp
(setq a (randn `(10 10)))
(with-instant-kernel a
    `(print ,a)) ;; -> (print a) isn't evaluated

(funcall (build *)) ;; -> (print a) will be evaluated.
```

Note that (equal (with-instant-kernel a) a) is NIL, that is, the returned value of this macro must be followed by a calculation node.

If the return value of Body can be expanded as a macro, the values are compiled together at JIT compile time. Otherwise, the given tensor is returned as is.


## declare-local-variables
TODO
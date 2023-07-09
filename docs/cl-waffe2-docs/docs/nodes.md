
# Formulate Neural Networks
The package `:cl-waffe2/vm.nodes` provides a fundamental system for building neural networks using `AbstractTensor`.

This package can be divided into three main parts.

1. Shaping APIs (`:where`)
2. defnode  (The smallest unit of differentiable operations)
3. defmodel (Operations consisted of defnode, and static functions)


## Shaping API


When defining an operation in cl-waffe2 with a `defnode` macro, the shape of the matrix used in the operation must also be defined in the `:where` keyword.

This is a Shaping API, and responsible for shape inspection of all operations.

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


## [macro] defnode


```lisp
(defnode ((abstract-name
		   (self &rest constructor-arguments)
		    &key
		      (where t)
		      (out-scalar-p nil)
                      (save-for-backward nil)
		      (slots nil)
		      (backward nil)
		      (documentation ""))
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
g(dout, dx_{in}, dy_{in}, ..., dn_{in}) \triangleq \\
 Move(dx_{out}, {dout} \times {dx_{grad}}),\\
 Move(dx_{out}, {dout} \times {dy_{grad}}),\\
 ...,\\
 Move(dx_{out}, {dout} \times {dn_{grad}})
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
\begin{equation}
  x_{in}=
  \begin{cases}
    x_{saveforbackward} & \text{SaveForBackward is t} \\
    \text{x} & \text{otherwise}
  \end{cases}
\end{equation}
```

```math
\begin{equation}
  x_{out}=
  \begin{cases}
    x_{copy} & \text{If the tensor is a ExistTensor or cause conflicts} \\
    \text{x} & \text{If the tensor make no conflicts.}
  \end{cases}
\end{equation}
```

(`x` is a variable called with `(forward node &rest inputs)` function.)


### Example

```lisp

(defnode (MatMulNode (myself dtype &key transpose-a transpose-b)
	  :where (A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :slots ((transpose-a :initarg :transpose-a :type boolean :reader trans-a?)
		  (transpose-b :initarg :transpose-b :type boolean :reader trans-b?))
          :documentation "gemm"
	  :backward ((self dout da db do)
		     (declare (ignore do))
		     (values
		      (!matmul dout (!t db))
		      (!matmul (!t da) dout)
		      nil))))

(MatmulNode :float)
;; <Node: MATMULNODE-CPUTENSOR (A[~ I J] B[~ J K] C[~ I K] -> C[~ I K])>
```


## [macro] define-impl

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

## [generic] forward
```(forward node &rest inputs)```
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
  :backward <Node: ADDNODE-CPUTENSOR (A[~] B[~] -> A[~])>}
```

## [class] defmodel

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

`defmodel` defines a new `Composite` class which describes network structures with using lazy-evaluated tensor. Viewing the set of `AbstractNode` as a single cohesive entity, you can formulate the forward propagation in `on-call->` keyword.

`Composite` is used as a `neural network model` if used as a merely data structure, but combined with `define-composite-function`, `Composite` can also define a single statically-operation function from a set of nodes.

A new `Composite` class is initialized with `(name &rest inputs)` function, being called with a `call` method.

### Effects

1. defines a class named **name**

2. defines a function named **name** with the constructor-arguments and constructor-body.

### Inputs

  1. `name[Symbol]` the macro defines an class and constructor function named after it.

  2. `(self-name &rest constructor-arguments)` An initializer form of `constructor function`.

  3. `slots ((slot-option1) (slot-option2) ...)` Parameters of the inherited Composite class. It has the same syntax as defclass slots

  4. `initargs (:accessor-name1 accessor-init-form1 :accessor-name2 accessor-init-form2 ...` Unlike structures, CLOS classes are somewhat more cumbersome to initialise. To make this simple, this argument was introduced. Describe here initializer form in advance.

  5. `documentation[String]`

  6. `on-call-> [One of: nil symbol-name function list]`
     on-call-> is used to control the behaviour of **call** function.

  7. `where[Subscript DSL] (Optional)` Describe the state of the Tensor before and after `on-call->`

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

;; Using Lazily...
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


;; Defines a statically working function.
(define-composite-function (Softmax-Model) !softmax-static)

(!softmax-static (randn `(10 10)))

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

### How to use on-call-> form?

In the keyword `on-call->`, describe the behaviour when called with a `call` function following this forms:

### `on-call->` = nil

In that case, cl-waffe2 calls the `call` method when doing forward propagation of the model.

### `on-call->` is a symbol-name

cl-waffe2 calls the function named `symbol-name`.

For example, setting `:on-call-> = call-example-layer` and defining a `call-example-layer` method.

```lisp
(defmethod call-example-layer ((model ExampleLayer) x y)
    (print "call-example-layer is used!"))
```

```lisp
(call (ExampleLayer 10) tensor) ;; call-example-layer is used!
```

### on-call-> is a function name or a lambda.

cl-waffe2 calls the given lambda function as a forward propagation.

### `on-call->` is a list


```lisp
(Example)
:on-call-> ((self x) (!sin x))
```

This argument is expanded into `#'(lambda ,@on-call->)` and works as well as 3.

## call


```lisp
(call model &rest inputs)
```

`call` is a generic function which is used to `:forward`/`:on-call->` forms for an `AbstractNode`/`Composite` class respectively.
## with-devices
The macro `with-devices` declares the priority of dispatching nodes.

### Input

1. `backend-priority` An list of device's name (e.g.: CPUTensor, LispTensor...) Devices on the left have higher priority.

### Example

Let `ATensor` and `BTensor` be a pointer compatible, and subclass of `AbstractTensor`, and operations defined is following:

1. ATensor has !add.
2. BTensor has !mul.

```lisp
(setq a (make-tensor `(10 10))) ;; The tensor a is ATensor.

;; (Priority1=ATensor Priority2=BTensor)
(with-devices (ATensor BTensor)
   (!add a (!mul a a)))
```

cl-waffe2's backend dispatching rule is following:

If the priority 1 backend does not have an implementation of the specified operation, check if the priority 2 backend does, if it still does not have it, 3, 4... and so on.

The order of priority would be `(,@backend-priority ScalarTensor t). (t is a special name, and it implys the implement works for all the backends.)


```lisp
(with-devices (LispTensor CPUTensor)
   (!add a b))
```
## [macro] define-and-impl-node

```lisp
(define-and-impl-node (abstract-name
				 (self &rest constructor-arguments)
				 &key
				   (device t)
				   (cache-when-compiled t)
				   (reject-p nil)
				   (where t)
				   (out-scalar-p nil)
				   (slots nil)
				   (save-for-backward nil)
				   (forward nil)
				   (backward nil)
				   (documentation "")))
```

Expands `defnode` and `define-impl` at the same time.

## [macro] define-composite-function

```lisp
(define-composite-function composite-init-form
		       	     function-name
	       		     &key
      			       (dtype t)
			       (order :column)
		       	       (compile-mode :default))
```

Tracing the `on-call->` form of a given composite-init-form, the macro `define-composite-function` defines a function of calling `on-call->` statically.

On the condition where composite should be defined as polymorphic, the function is also defined as generic definition/dispatching, otherwise, defines as a single defun form.

### Inputs

1. `composite-init-form` Set here an initform of `Composite`, to be traced.

2. `function-name` the compiled function is defined as this name.

3. `:dtype[boolean or keyword]` Set t to make compiled function work on any dtypes, or set `keyword` to use.

4. `order[keyword]` Element major.

5. `compile-mode[compile-mode-t]` compiling option.

## [class] Composite

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

## [class] AbstractNode

[class] AbstractNode

The class AbstractNode is a fundamental object of describing computation nodes in cl-waffe.

AbstractNode must possess following:

   1. Transimission State

   2. Slots (for passing forward/backward)

   3. Variables (for building computation nodes)

## with-instant-kernel

```lisp
(with-instant-kernel tensor &body body)
```

Continues the computation node following tensor with embedding an `instant-kernel`. `Instant` is Lisp code that can be embedded in compiled functions.

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

Note that `(equal (with-instant-kernel a) a)` is `NIL`, that is, the returned value of this macro must be followed by a calculation node.

If the return value of `body` can be expanded as a macro, the values are compiled together at JIT compile time. Otherwise, the given tensor is returned as is.


# cl-waffe2 VM

The package `cl-waffe2/vm` is the central of system, and features are focused on low-level stuffs: compiling/optimizing/rewriting cl-waffe2 IRs and how they're executed. So, most of APIs are accesible by convinient API wrappers of other packages.

- Global Variables
    - [optimization level](./#parameter-opt-level)
    - [logging](./#parameter-logging-vm-execution)
- IR and Compiler
    - [WfInstruction](./#struct-wfinstruction)
    - [compiler](./#function-compile-forward-and-backward)
    - [acceptor](./#function-accept-instructions)
- Analyzing compiled codes
    - [disassemble](#function-disassemble-waffe2-ir)
    - [profiling](#function-benchmark-accept-instructions)
- Adding Symbolic Diff and Device-Specific Optimization
    - [FusionPathQuery](./#struct-fusionpathquery)
    - [defpath](./#macro-defpath)
## [parameter] `*opt-level*`

This parameter indicates the degree of runtime error detection. Whichever you choose, cl-waffe2 never apply something unsafe code transformation. It takes the fixnum from 1 to 3, and the larger, the faster.

- Set 1 to use safety-mode, in every instructions, runtime error is checked.

- Set 2 to use middle-mode, runtime error is checked only when first execution.

- Set 3 to use fastest-mode, no runtime error checking is done.

Again, whichever levels you choose, the graph cl-waffe2 executes is the same. So the effects on the performance is very small (within < `1e-4~1e-5` sec).

In default, set to 2.

## [parameter] `*logging-vm-execution*`

This parameter is useful for printing how all instructions are performed. If set to T, all results and arguments produced by executing `cl-waffe2 IR` is displayed into the terminal. In default, set to nil.

## [struct] WfInstruction

WfInstruction is IR for cl-waffe2 VM. Its form is represented by an extended Wengert list which is able to return multiple outputs. In this document, we call this *cl-waffe2 IR*, and compiled cl-waffe2 code, that is, the list of WfInstruction is called **InstructionSeq** or **iseq**. Unlike other frameworks, this IR is not only used to represent backpropagation but also forward propagation.


In cl-waffe2, WfInstruction is created by compiling AbstractNode, and its operation can be obtained by compiling lisp code or passing lambda functions.

A single WfInstruction represents:

```
out_to[0], out_to[1], ... <- λ(Args[0], Args[1], Args[2], ...)
 ^wfop-out-to                 ^wfop-op     ^wfop-args
```

where λ represents the operation. And, if any, `ArgsN` is wrapped with `SV4BW`.

```
out_to[0], out_to[1], ... <- λ(SV4BW(Args[0]), Args[1], Args[2], ...)
                                  ^ wfop-sv4bw
```

SV4BW (i.e: save-for-backward) is a temporary tensor to compute backwards and cl-waffe2 reads the `:save-for-backward` slots in the `define-impl` macro, and the corresponding tensors are copied.

### Slots

`wfop-op[function]` corresponds with compiled λ function.

`wfop-node[AbstractNode or string or function]` The node which generates λ function. For the most case, this slot is set to `AbstractNode`, but the node is something special, (e.g.: `CodeBlock`, `IfNode` etc...), set to `function`.

`wfop-out-to[list of AbstractTensor]` indicates list of tensors that results are to be stored.

`wfop-self[AbstractTensor]` corresponds with `out_target`, that is, the tensor to store the results

`wfop-args[list of AbstractTensor]` corresponds with `(tensor-variable wfop-self)`. tensors to be called with: `arg1 arg2 arg3...`.

`wfop-sv4bw[list of AbstractTensor]` indicates list of tensors storing save-for-backward tensors. if the corresponding position is `save-for-backward=nil`, the corresponding position also become nil.


## [function] compile-forward-and-backward

```lisp
(compile-forward-and-backward toplevel &key (need-backward t) (fuse-p t) (compile-mode :default) (optimize-locality t))
```

Compiles into cl-waffe2 IR (so-called iseq) from the given toplevel to each leaf points (where detach-p=t or backward=null variables). `toplevel` is AbstractTensor with backwards.

Tips: `disassemble-waffe2-ir` to display compiled Instruction Sequence.

### Return

`(values forward-iseq backward-iseq leaves[an list of AbstractTensor that appeared in the node] dout alloc-state)`

## [function] accept-instructions

```lisp
(accept-instructions iseq)
```

Evaluates generated cl-waffe2 IR sequence.

`iseq[list]` an list of `WFInstruction`

## [function] disassemble-waffe2-ir

```lisp
(disassemble-waffe2-ir toplevel &key (backward t) (stream t) (fuse-p t))
```

Prints out the compiled cl-waffe2 IR from toplevel to each leaf points to `stream`. If `backward` was set to t, `backward` is also displayed.

### Example

```lisp
(with-output-to-string (out)
    (disassemble-waffe2-ir (!softmax (parameter (randn `(3 3))) :avoid-overflow nil) :stream out))


disassemble-waffe2-ir:
 [Forward]: 
<WfInst[op=ALLOC{INTERNAL}]     : TID341 <= op(TID341{float, (3 3)} <Param>TID336{float, (3 3)})>
<WfInst[op=EXPNODE-CPUTENSOR]   : TID341 <= op(<Param>SV4BW(TID336{float, (3 3)}) TID341{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR] : TID371 <= op(TID371{float, (3 1)} <Input>TID363{float, (1)})>
<WfInst[op=VIEWTENSORNODE-T]    : TID371 <= op(TID371{float, (3 3)} TID371{float, (3 1)})>
<WfInst[op=ADDNODE-CPUTENSOR]   : TID371 <= op(TID371{float, (3 3)} TID341{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]   : TID341 <= op(SV4BW(TID341{float, (3 3)}) SV4BW(TID371{float, (3 3)}))>

6 Instructions | 3 Tensors | 1 Scalars


 [Pullback]: 
<WfInst[op=MOVETENSORNODE-CPUTENSOR]       : TID430 <= op(TID430{float, (3 3)} <Input>TID427{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]              : TID430 <= op(TID430{float, (3 3)} TID408{float, (3 3)})>
<WfInst[op=MOVETENSORNODE-CPUTENSOR]       : TID458 <= op(TID458{float, (3 3)} <Input>TID427{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR]            : TID458 <= op(TID458{float, (3 3)} <Input>TID455{float, (1)})>
<WfInst[op=MULNODE-CPUTENSOR]              : TID403 <= op(TID403{float, (3 3)} TID458{float, (3 3)})>
<WfInst[op=VIEWTENSORNODE-T]               : TID408 <= op(TID408{float, (3 3)} TID408{float, (3 1)})>
<WfInst[op=MULNODE-CPUTENSOR]              : TID408 <= op(TID408{float, (3 3)} TID408{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]              : TID403 <= op(TID403{float, (3 3)} TID408{float, (3 3)})>
<WfInst[op=SYSTEM-LAZY-CONS-T]             : TID430 TID403 <= op(TID430{float, (3 3)} TID403{float, (3 3)})>
<WfInst[op=EXPNODE-CPUTENSOR]              : TID351 <= op(TID351{float, (3 3)} TID351{float, (3 3)})>
<WfInst[op=MULNODE-CPUTENSOR]              : TID430 <= op(TID430{float, (3 3)} TID351{float, (3 3)})>
<WfInst[op={GRAD}MOVETENSORNODE-CPUTENSOR] : <Input>TID338 <= op(<Input>TID338{float, (3 3)} TID430{float, (3 3)})>

12 Instructions | 7 Tensors | 1 Scalars


```

## [function] benchmark-accept-instructions

```lisp
(benchmark-accept-instructions iseq &key (n-sample 1) (ignore-first-call nil) (stream t) (top-k 10))
```

Basically, the function `benchmark-accept-instruction` executes the given list of instructions with profiling execution time, but at the end of proess, displays the report into `stream`.

### Inputs

`n-sample[fixnum]` repeats the iseq execution for `n-sample` times

`ignore-first-call[boolean]` If t, ignores the first call to avoid including allocating time.

`stream[stream]` the place to display the result

`top-k[fixnum]` top-k slowest nodes are displayed at the end of report.

### Return

`result[AbstractTensor]`

See also: `proceed-bench`

### Example

```lisp
(with-output-to-string (out)
    (proceed-bench (!softmax (randn `(100 100))) :n-sample 100 :stream out))

[Sorted by Instructions]
 Time(s)   |   Instruction ( * - Beyonds the average execution time)
4.83e-4    | <WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID693 <= op(TID693{float, (100 100)} <Input>TID609{float, (100 100)})>
1.32e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID687 <= op(TID687{float, (100 1)} TID687{float, (100 1)})>
2.55e-4    | <WfInst[op=SCALARMUL-CPUTENSOR]      : TID687 <= op(TID687{float, (100 1)} <Input>TID618{float, (1)})>
1.24e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID687 <= op(TID687{float, (100 100)} TID687{float, (100 1)})>
0.007934*  | <WfInst[op=ADDNODE-CPUTENSOR]        : TID687 <= op(TID687{float, (100 100)} <Input>TID609{float, (100 100)})>
1.44e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID687 <= op(TID687{float, (100 1)} TID687{float, (100 100)})>
0.00277*   | <WfInst[op=SCALARDIV-CPUTENSOR]      : TID687 <= op(TID687{float, (100 1)} <Input>TID613{float, (1)})>
1.27e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID687 <= op(TID687{float, (100 100)} TID687{float, (100 1)})>
0.005669*  | <WfInst[op=SUBNODE-CPUTENSOR]        : TID693 <= op(TID693{float, (100 100)} TID687{float, (100 100)})>
0.001311   | <WfInst[op=EXPNODE-CPUTENSOR]        : TID693 <= op(TID693{float, (100 100)} TID693{float, (100 100)})>
2.88e-4    | <WfInst[op=SCALARMUL-CPUTENSOR]      : TID687 <= op(TID687{float, (100 1)} <Input>TID738{float, (1)})>
1.24e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID687 <= op(TID687{float, (100 100)} TID687{float, (100 1)})>
0.007932*  | <WfInst[op=ADDNODE-CPUTENSOR]        : TID687 <= op(TID687{float, (100 100)} TID693{float, (100 100)})>
0.006103*  | <WfInst[op=DIVNODE-CPUTENSOR]        : TID693 <= op(TID693{float, (100 100)} TID687{float, (100 100)})>

14 Instructions | 6 Tensors | Overheads due to SV4BW(...) -> 6.96e-6(s) 

 Total Time: 0.033396002 sec

[Sorted by topK]
 Instruction                         | Total time (s) | Time/Total (n-sample=100)
<WfInst[op=ADDNODE-CPUTENSOR]        | 0.015866       | 47.508682%
<WfInst[op=DIVNODE-CPUTENSOR]        | 0.006103       | 18.274643%
<WfInst[op=SUBNODE-CPUTENSOR]        | 0.005669       | 16.975086%
<WfInst[op=SCALARDIV-CPUTENSOR]      | 0.00277        | 8.294406%
<WfInst[op=EXPNODE-CPUTENSOR]        | 0.001311       | 3.9256196%
<WfInst[op=VIEWTENSORNODE-T]         | 6.5100007e-4   | 1.9493352%
<WfInst[op=SCALARMUL-CPUTENSOR]      | 5.43e-4        | 1.6259432%
<WfInst[op=MOVETENSORNODE-CPUTENSOR] | 4.83e-4        | 1.446281%

```

## [struct] FusionPathQuery

```lisp
(make-query abstract-node &key (device t) (dtype t) (pred #'(lambda (node) t)))
```

`(make-query ...)` and create a new query.

A single `FusionPathQuery` becomes t only when satisfies all of following conditions:

`abstract-node[symbol]` become t when the node is a subtype of `abstract-node`

`device[t or symbol]`   become t when the node is working under the device or `subtype` of it.

`dtype[t or list]`      become t when the `dtype` is set to t, or the list of dtype in arguments are corresponds with the list. (e.g.: `(list :float :float)`)

`pred[function]`        specifies an additional predicator, the function receives `(node)` as arguments and return t to accept it. (`arguments-tensor` is an list of tensors, which `forward` or `call` used.)

This structure is excepted to be combined with `defpath`.

## [macro] defpath

```lisp
(defpath (fusion-name &rest query-list) &key (reject-p #'(lambda ())) (replaced-with nil))
```

⚠️ This API is still in the conceptial stage, tests are not enough. DO NOT USE THIS.

The macro defpath introduces to cl-waffe2 **Symbolic Differentiation**. Users can define a `FusionQueryPath` to relocate compiled instructions with reference to the search. Composing the sequence of generated IRs to suit the device or model is the easiest way to speed up your model, cl-waffe2 searches for compiled nodes and replaces those matching the conditions specified in `query-list` with the computed nodes specified in `replaced-with`, if `:fuse-p` is set to t (default: `t`). In the simplest case, `defpath` can detect `[AddNode-CPUTensor] [MulNode-CPUTensor]` sequence and replace it with `[AddMulNode-CPUTensor]` node to reduce the number of instructions.

```lisp
[When adding a new device to cl-waffe2...]
 1. Declare the new device (e.g.: CPUTensor, CUDATensor ...)
 2. Prepare allocator and accessors (e.g.: initialize-instance method, vref and (setf vref))
 3. Implement existing operations with define-impl macro
 4. Blush up the generated IR with defpath macro to fuse more operations in a small cycle. <- defpath, here!
```

The created and registered path, will be reset with the `(reset-all-path!)` function. All registered paths are stored in `*user-defined-path-list*` parameter.

### Rules

cl-waffe2 replaces the existing operations with following the rules:

1. The search is performed ignoring SaveForBackwardNode. If it is contained in the area to be replaced, it is moved to the last sequence of replaced one.


```lisp
;; Example
Rule: [A] [B] -> [C]
```

```lisp
Before Fusion:

[A]
[SAVE_FOR_BACKWARD]
[B]
[M]
[N]
```

```lisp
Searching will be done ignoring [SAVE_FOR_BACKWARD]

^ [A]
| [B]
| [M]
| [N]
reading in this direction.
```

```lisp
After fusion:

[C]  ;; [A] [B] -> [C]
[SAVE_FOR_BACKWARD] ;; placed after the operation
[M]
[N]
```

2. `defpath` priority is given to those registered first.

Repeat the search until no more targets are found to replace it.

3. query-list

Not replaced until the `query-list` matches everything, including the order.

### Example

(TODO: For the case of ReLU)

### make-query



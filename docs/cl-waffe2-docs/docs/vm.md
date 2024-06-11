
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

`wfop-comment[string or null]` If any, the value of slot is displayed when printing IR.

`wfop-loadp[boolean]` If set to t, the operation is interpreted as `load-pointer`. See also: (read-loadp instruction)

`wfop-lut-cache-p[boolean]` If set to T, indicates `wfop-op` is already compiled and cached in LUT.


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
<WfInst[op=ALLOC{INTERNAL}]    : TID398 <= op(TID398{float, (3 3)} <Param>TID393{float, (3 3)})>
<WfInst[op=EXPNODE-LISPTENSOR] : TID398 <= op(<Param>SV4BW(TID393{float, (3 3)}) TID398{float, (3 3)})>
<WfInst[load_pointer{SYS}]     : TID438* = TID438*>
<WfInst[op=MULNODE-LISPTENSOR] : TID456 <= op(TID456{float, (3 1)} <Input>TID438{float, (3 1)})>
<WfInst[load_pointer{SYS}]     : TID456* = TID456*>
<WfInst[op=ADDNODE-LISPTENSOR] : TID456 <= op(TID456{float, (3 3)} TID398{float, (3 3)})>
<WfInst[op=DIVNODE-LISPTENSOR] : TID398 <= op(SV4BW(TID398{float, (3 3)}) SV4BW(TID456{float, (3 3)}))>

7 Instructions | 4 Tensors | 0 Scalars


 [Pullback]: 
<WfInst[op=MOVETENSORNODE-LISPTENSOR] : TID506 <= op(TID506{float, (3 3)} <Input>TID503{float, (3 3)})>
<WfInst[op=DIVNODE-LISPTENSOR]        : TID506 <= op(TID506{float, (3 3)} TID488{float, (3 3)})>
<WfInst[op=MOVETENSORNODE-LISPTENSOR] : TID521 <= op(TID521{float, (3 3)} <Input>TID503{float, (3 3)})>
<WfInst[load_pointer{SYS}]            : TID530* = TID530*>
<WfInst[op=MULNODE-LISPTENSOR]        : TID521 <= op(TID521{float, (3 3)} <Input>TID530{float, (3 3)})>
<WfInst[op=MULNODE-LISPTENSOR]        : TID521 <= op(TID521{float, (3 3)} TID483{float, (3 3)})>
<WfInst[load_pointer{SYS}]            : TID488* = TID488*>
<WfInst[op=MULNODE-LISPTENSOR]        : TID488 <= op(TID488{float, (3 3)} TID488{float, (3 3)})>
<WfInst[op=DIVNODE-LISPTENSOR]        : TID521 <= op(TID521{float, (3 3)} TID488{float, (3 3)})>
<WfInst[op=EXPNODE-LISPTENSOR]        : TID418 <= op(TID418{float, (3 3)} TID418{float, (3 3)})>
<WfInst[op=MULNODE-LISPTENSOR]        : TID506 <= op(TID506{float, (3 3)} TID418{float, (3 3)})>
<WfInst[load_pointer{SYS}]            : TID395* = TID506*>

12 Instructions | 8 Tensors | 0 Scalars


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
0.001247   | <WfInst[op=MOVETENSORNODE-LISPTENSOR] : TID715 <= op(TID715{float, (100 100)} <Input>TID643{float, (100 100)})>
1.3e-5     | <WfInst[load_pointer{SYS}]            : TID656* = TID656*>
9.04e-4    | <WfInst[op=MULNODE-LISPTENSOR]        : TID709 <= op(TID709{float, (100 1)} <Input>TID656{float, (100 1)})>
3.9e-5     | <WfInst[load_pointer{SYS}]            : TID709* = TID709*>
0.003847*  | <WfInst[op=ADDNODE-LISPTENSOR]        : TID709 <= op(TID709{float, (100 100)} <Input>TID643{float, (100 100)})>
4.2e-5     | <WfInst[load_pointer{SYS}]            : TID709* = TID709*>
2.8e-5     | <WfInst[load_pointer{SYS}]            : TID697* = TID697*>
9.78e-4    | <WfInst[op=DIVNODE-LISPTENSOR]        : TID709 <= op(TID709{float, (100 1)} <Input>TID697{float, (100 1)})>
3.6e-5     | <WfInst[load_pointer{SYS}]            : TID709* = TID709*>
0.003052*  | <WfInst[op=SUBNODE-LISPTENSOR]        : TID715 <= op(TID715{float, (100 100)} TID709{float, (100 100)})>
0.004237*  | <WfInst[op=EXPNODE-LISPTENSOR]        : TID715 <= op(TID715{float, (100 100)} TID715{float, (100 100)})>
2.2e-5     | <WfInst[load_pointer{SYS}]            : TID755* = TID755*>
9.04e-4    | <WfInst[op=MULNODE-LISPTENSOR]        : TID709 <= op(TID709{float, (100 1)} <Input>TID755{float, (100 1)})>
4.6e-5     | <WfInst[load_pointer{SYS}]            : TID709* = TID709*>
0.004345*  | <WfInst[op=ADDNODE-LISPTENSOR]        : TID709 <= op(TID709{float, (100 100)} TID715{float, (100 100)})>
0.003068*  | <WfInst[op=DIVNODE-LISPTENSOR]        : TID715 <= op(TID715{float, (100 100)} TID709{float, (100 100)})>

16 Instructions | 6 Tensors | Overheads due to SV4BW(...) -> 2.26e-6(s) 

 Total Time: 0.022808 sec

[Sorted by topK]
 Instruction                          | Total time (s) | Time/Total (n-sample=100)
<WfInst[op=ADDNODE-LISPTENSOR]        | 0.008192       | 35.91722%
<WfInst[op=EXPNODE-LISPTENSOR]        | 0.004237       | 18.576815%
<WfInst[op=DIVNODE-LISPTENSOR]        | 0.004046       | 17.73939%
<WfInst[op=SUBNODE-LISPTENSOR]        | 0.003052       | 13.381269%
<WfInst[op=MULNODE-LISPTENSOR]        | 0.001808       | 7.927043%
<WfInst[op=MOVETENSORNODE-LISPTENSOR] | 0.001247       | 5.4673796%
<WfInst[op=VIEWTENSORNODE-T]          | 2.26e-4        | 0.99088037%

```

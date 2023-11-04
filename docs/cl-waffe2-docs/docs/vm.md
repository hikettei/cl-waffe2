
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
<WfInst[op=ALLOC{INTERNAL}]     : TID474 <= op(TID474{float, (3 3)} <Param>TID469{float, (3 3)})>
<WfInst[op=EXPNODE-CPUTENSOR]   : TID474 <= op(<Param>SV4BW(TID469{float, (3 3)}) TID474{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR] : TID513 <= op(TID513{float, (3 1)} <Input>TID502{float, (1)})>
<WfInst[load_pointer{SYS}]      : TID513* = TID513*>
<WfInst[op=ADDNODE-CPUTENSOR]   : TID513 <= op(TID513{float, (3 3)} TID474{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]   : TID474 <= op(SV4BW(TID474{float, (3 3)}) SV4BW(TID513{float, (3 3)}))>

6 Instructions | 3 Tensors | 1 Scalars


 [Pullback]: 
<WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID557 <= op(TID557{float, (3 3)} <Input>TID554{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]        : TID557 <= op(TID557{float, (3 3)} TID543{float, (3 3)})>
<WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID578 <= op(TID578{float, (3 3)} <Input>TID554{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR]      : TID578 <= op(TID578{float, (3 3)} <Input>TID575{float, (1)})>
<WfInst[op=MULNODE-CPUTENSOR]        : TID538 <= op(TID538{float, (3 3)} TID578{float, (3 3)})>
<WfInst[load_pointer{SYS}]           : TID543* = TID543*>
<WfInst[op=MULNODE-CPUTENSOR]        : TID543 <= op(TID543{float, (3 3)} TID543{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]        : TID538 <= op(TID538{float, (3 3)} TID543{float, (3 3)})>
<WfInst[op=EXPNODE-CPUTENSOR]        : TID490 <= op(TID490{float, (3 3)} TID490{float, (3 3)})>
<WfInst[op=MULNODE-CPUTENSOR]        : TID557 <= op(TID557{float, (3 3)} TID490{float, (3 3)})>
<WfInst[load_pointer{SYS}]           : TID471* = TID557*>

11 Instructions | 7 Tensors | 1 Scalars


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
3.53e-4    | <WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID801 <= op(TID801{float, (100 100)} <Input>TID735{float, (100 100)})>
1.62e-4    | <WfInst[op=SCALARMUL-CPUTENSOR]      : TID795 <= op(TID795{float, (100 1)} <Input>TID744{float, (1)})>
9.2e-5     | <WfInst[load_pointer{SYS}]           : TID795* = TID795*>
0.006241*  | <WfInst[op=ADDNODE-CPUTENSOR]        : TID795 <= op(TID795{float, (100 100)} <Input>TID735{float, (100 100)})>
9.3e-5     | <WfInst[load_pointer{SYS}]           : TID795* = TID795*>
1.52e-4    | <WfInst[op=SCALARDIV-CPUTENSOR]      : TID795 <= op(TID795{float, (100 1)} <Input>TID739{float, (1)})>
9.3e-5     | <WfInst[load_pointer{SYS}]           : TID795* = TID795*>
0.004291*  | <WfInst[op=SUBNODE-CPUTENSOR]        : TID801 <= op(TID801{float, (100 100)} TID795{float, (100 100)})>
9.85e-4    | <WfInst[op=EXPNODE-CPUTENSOR]        : TID801 <= op(TID801{float, (100 100)} TID801{float, (100 100)})>
1.55e-4    | <WfInst[op=SCALARMUL-CPUTENSOR]      : TID795 <= op(TID795{float, (100 1)} <Input>TID839{float, (1)})>
9.7e-5     | <WfInst[load_pointer{SYS}]           : TID795* = TID795*>
0.006674*  | <WfInst[op=ADDNODE-CPUTENSOR]        : TID795 <= op(TID795{float, (100 100)} TID801{float, (100 100)})>
0.004137*  | <WfInst[op=DIVNODE-CPUTENSOR]        : TID801 <= op(TID801{float, (100 100)} TID795{float, (100 100)})>

13 Instructions | 6 Tensors | Overheads due to SV4BW(...) -> 5.03e-6(s) 

 Total Time: 0.023525001 sec

[Sorted by topK]
 Instruction                         | Total time (s) | Time/Total (n-sample=100)
<WfInst[op=ADDNODE-CPUTENSOR]        | 0.012915       | 54.899044%
<WfInst[op=SUBNODE-CPUTENSOR]        | 0.004291       | 18.24017%
<WfInst[op=DIVNODE-CPUTENSOR]        | 0.004137       | 17.585546%
<WfInst[op=EXPNODE-CPUTENSOR]        | 9.85e-4        | 4.187035%
<WfInst[op=VIEWTENSORNODE-T]         | 3.75e-4        | 1.5940487%
<WfInst[op=MOVETENSORNODE-CPUTENSOR] | 3.53e-4        | 1.5005313%
<WfInst[op=SCALARMUL-CPUTENSOR]      | 3.17e-4        | 1.3475026%
<WfInst[op=SCALARDIV-CPUTENSOR]      | 1.52e-4        | 0.6461211%

```

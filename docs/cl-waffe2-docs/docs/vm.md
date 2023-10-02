
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
<WfInst[op=ALLOC{INTERNAL}]     : TID520 <= op(TID520{float, (3 3)} <Param>TID515{float, (3 3)})>
<WfInst[op=EXPNODE-CPUTENSOR]   : TID520 <= op(<Param>SV4BW(TID515{float, (3 3)}) TID520{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR] : TID550 <= op(TID550{float, (3 1)} <Input>TID542{float, (1)})>
<WfInst[op=VIEWTENSORNODE-T]    : TID550 <= op(TID550{float, (3 3)} TID550{float, (3 1)})>
<WfInst[op=ADDNODE-CPUTENSOR]   : TID550 <= op(TID550{float, (3 3)} TID520{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]   : TID520 <= op(SV4BW(TID520{float, (3 3)}) SV4BW(TID550{float, (3 3)}))>

6 Instructions | 3 Tensors | 1 Scalars


 [Pullback]: 
<WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID608 <= op(TID608{float, (3 3)} <Input>TID605{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]        : TID608 <= op(TID608{float, (3 3)} TID587{float, (3 3)})>
<WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID636 <= op(TID636{float, (3 3)} <Input>TID605{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR]      : TID636 <= op(TID636{float, (3 3)} <Input>TID633{float, (1)})>
<WfInst[op=MULNODE-CPUTENSOR]        : TID582 <= op(TID582{float, (3 3)} TID636{float, (3 3)})>
<WfInst[op=VIEWTENSORNODE-T]         : TID587 <= op(TID587{float, (3 3)} TID587{float, (3 1)})>
<WfInst[op=MULNODE-CPUTENSOR]        : TID587 <= op(TID587{float, (3 3)} TID587{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]        : TID582 <= op(TID582{float, (3 3)} TID587{float, (3 3)})>
<WfInst[op=SYSTEM-LAZY-CONS-T]       : TID608 TID582 <= op(TID608{float, (3 3)} TID582{float, (3 3)})>
<WfInst[op=EXPNODE-CPUTENSOR]        : TID530 <= op(TID530{float, (3 3)} TID530{float, (3 3)})>
<WfInst[op=MULNODE-CPUTENSOR]        : TID608 <= op(TID608{float, (3 3)} TID530{float, (3 3)})>
<WfInst[op={GRAD}SETQ{INTERNAL}]     : <Input>TID517 <= op(<Input>TID517{float, (3 3)} TID608{float, (3 3)})>

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
3.45e-4    | <WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID885 <= op(TID885{float, (100 100)} <Input>TID801{float, (100 100)})>
9.2e-5     | <WfInst[op=VIEWTENSORNODE-T]         : TID879 <= op(TID879{float, (100 1)} TID879{float, (100 1)})>
1.37e-4    | <WfInst[op=SCALARMUL-CPUTENSOR]      : TID879 <= op(TID879{float, (100 1)} <Input>TID810{float, (1)})>
9.6e-5     | <WfInst[op=VIEWTENSORNODE-T]         : TID879 <= op(TID879{float, (100 100)} TID879{float, (100 1)})>
0.006066*  | <WfInst[op=ADDNODE-CPUTENSOR]        : TID879 <= op(TID879{float, (100 100)} <Input>TID801{float, (100 100)})>
9.4e-5     | <WfInst[op=VIEWTENSORNODE-T]         : TID879 <= op(TID879{float, (100 1)} TID879{float, (100 100)})>
0.001993*  | <WfInst[op=SCALARDIV-CPUTENSOR]      : TID879 <= op(TID879{float, (100 1)} <Input>TID805{float, (1)})>
9.4e-5     | <WfInst[op=VIEWTENSORNODE-T]         : TID879 <= op(TID879{float, (100 100)} TID879{float, (100 1)})>
0.004204*  | <WfInst[op=SUBNODE-CPUTENSOR]        : TID885 <= op(TID885{float, (100 100)} TID879{float, (100 100)})>
9.67e-4    | <WfInst[op=EXPNODE-CPUTENSOR]        : TID885 <= op(TID885{float, (100 100)} TID885{float, (100 100)})>
1.47e-4    | <WfInst[op=SCALARMUL-CPUTENSOR]      : TID879 <= op(TID879{float, (100 1)} <Input>TID930{float, (1)})>
1.01e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID879 <= op(TID879{float, (100 100)} TID879{float, (100 1)})>
0.005937*  | <WfInst[op=ADDNODE-CPUTENSOR]        : TID879 <= op(TID879{float, (100 100)} TID885{float, (100 100)})>
0.003947*  | <WfInst[op=DIVNODE-CPUTENSOR]        : TID885 <= op(TID885{float, (100 100)} TID879{float, (100 100)})>

14 Instructions | 6 Tensors | Overheads due to SV4BW(...) -> 4.95e-6(s) 

 Total Time: 0.024220001 sec

[Sorted by topK]
 Instruction                         | Total time (s) | Time/Total (n-sample=100)
<WfInst[op=ADDNODE-CPUTENSOR]        | 0.012003       | 49.558212%
<WfInst[op=SUBNODE-CPUTENSOR]        | 0.004204       | 17.357555%
<WfInst[op=DIVNODE-CPUTENSOR]        | 0.003947       | 16.29645%
<WfInst[op=SCALARDIV-CPUTENSOR]      | 0.001993       | 8.228736%
<WfInst[op=EXPNODE-CPUTENSOR]        | 9.67e-4        | 3.992568%
<WfInst[op=VIEWTENSORNODE-T]         | 4.7700002e-4   | 1.9694468%
<WfInst[op=MOVETENSORNODE-CPUTENSOR] | 3.45e-4        | 1.4244426%
<WfInst[op=SCALARMUL-CPUTENSOR]      | 2.84e-4        | 1.1725847%

```

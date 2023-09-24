
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
<WfInst[op=ALLOC{INTERNAL}]     : TID322 <= op(TID322{float, (3 3)} <Param>TID317{float, (3 3)})>
<WfInst[op=EXPNODE-CPUTENSOR]   : TID322 <= op(<Param>SV4BW(TID317{float, (3 3)}) TID322{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR] : TID352 <= op(TID352{float, (3 1)} <Input>TID344{float, (1)})>
<WfInst[op=VIEWTENSORNODE-T]    : TID352 <= op(TID352{float, (3 3)} TID352{float, (3 1)})>
<WfInst[op=ADDNODE-CPUTENSOR]   : TID352 <= op(TID352{float, (3 3)} TID322{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]   : TID322 <= op(SV4BW(TID322{float, (3 3)}) SV4BW(TID352{float, (3 3)}))>

6 Instructions | 3 Tensors | 1 Scalars


 [Pullback]: 
<WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID411 <= op(TID411{float, (3 3)} <Input>TID408{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]        : TID411 <= op(TID411{float, (3 3)} TID389{float, (3 3)})>
<WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID439 <= op(TID439{float, (3 3)} <Input>TID408{float, (3 3)})>
<WfInst[op=SCALARMUL-CPUTENSOR]      : TID439 <= op(TID439{float, (3 3)} <Input>TID436{float, (1)})>
<WfInst[op=MULNODE-CPUTENSOR]        : TID384 <= op(TID384{float, (3 3)} TID439{float, (3 3)})>
<WfInst[op=VIEWTENSORNODE-T]         : TID389 <= op(TID389{float, (3 3)} TID389{float, (3 1)})>
<WfInst[op=MULNODE-CPUTENSOR]        : TID389 <= op(TID389{float, (3 3)} TID389{float, (3 3)})>
<WfInst[op=DIVNODE-CPUTENSOR]        : TID384 <= op(TID384{float, (3 3)} TID389{float, (3 3)})>
<WfInst[op=SYSTEM-LAZY-CONS-T]       : TID411 TID384 <= op(TID411{float, (3 3)} TID384{float, (3 3)})>
<WfInst[op=EXPNODE-CPUTENSOR]        : TID332 <= op(TID332{float, (3 3)} TID332{float, (3 3)})>
<WfInst[op=MULNODE-CPUTENSOR]        : TID411 <= op(TID411{float, (3 3)} TID332{float, (3 3)})>
<WfInst[op={GRAD}SETQ{INTERNAL}]     : <Input>TID319 <= op(<Input>TID319{float, (3 3)} TID411{float, (3 3)})>

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
7.48e-4    | <WfInst[op=MOVETENSORNODE-CPUTENSOR] : TID674 <= op(TID674{float, (100 100)} <Input>TID590{float, (100 100)})>
2.53e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID668 <= op(TID668{float, (100 1)} TID668{float, (100 1)})>
3.58e-4    | <WfInst[op=SCALARMUL-CPUTENSOR]      : TID668 <= op(TID668{float, (100 1)} <Input>TID599{float, (1)})>
1.29e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID668 <= op(TID668{float, (100 100)} TID668{float, (100 1)})>
0.009711*  | <WfInst[op=ADDNODE-CPUTENSOR]        : TID668 <= op(TID668{float, (100 100)} <Input>TID590{float, (100 100)})>
1.53e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID668 <= op(TID668{float, (100 1)} TID668{float, (100 100)})>
0.004299*  | <WfInst[op=SCALARDIV-CPUTENSOR]      : TID668 <= op(TID668{float, (100 1)} <Input>TID594{float, (1)})>
1.66e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID668 <= op(TID668{float, (100 100)} TID668{float, (100 1)})>
0.008184*  | <WfInst[op=SUBNODE-CPUTENSOR]        : TID674 <= op(TID674{float, (100 100)} TID668{float, (100 100)})>
0.001539   | <WfInst[op=EXPNODE-CPUTENSOR]        : TID674 <= op(TID674{float, (100 100)} TID674{float, (100 100)})>
2.7e-4     | <WfInst[op=SCALARMUL-CPUTENSOR]      : TID668 <= op(TID668{float, (100 1)} <Input>TID719{float, (1)})>
1.27e-4    | <WfInst[op=VIEWTENSORNODE-T]         : TID668 <= op(TID668{float, (100 100)} TID668{float, (100 1)})>
0.009383*  | <WfInst[op=ADDNODE-CPUTENSOR]        : TID668 <= op(TID668{float, (100 100)} TID674{float, (100 100)})>
0.012125*  | <WfInst[op=DIVNODE-CPUTENSOR]        : TID674 <= op(TID674{float, (100 100)} TID668{float, (100 100)})>

14 Instructions | 6 Tensors | Overheads due to SV4BW(...) -> 1.036e-5(s) 

 Total Time: 0.047445003 sec

[Sorted by topK]
 Instruction                         | Total time (s) | Time/Total (n-sample=100)
<WfInst[op=ADDNODE-CPUTENSOR]        | 0.019094002    | 40.24449%
<WfInst[op=DIVNODE-CPUTENSOR]        | 0.012125       | 25.555906%
<WfInst[op=SUBNODE-CPUTENSOR]        | 0.008184       | 17.249445%
<WfInst[op=SCALARDIV-CPUTENSOR]      | 0.004299       | 9.061017%
<WfInst[op=EXPNODE-CPUTENSOR]        | 0.001539       | 3.2437556%
<WfInst[op=VIEWTENSORNODE-T]         | 8.28e-4        | 1.7451786%
<WfInst[op=MOVETENSORNODE-CPUTENSOR] | 7.48e-4        | 1.5765622%
<WfInst[op=SCALARMUL-CPUTENSOR]      | 6.28e-4        | 1.3236378%

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



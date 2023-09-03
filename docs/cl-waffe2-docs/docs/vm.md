
# cl-waffe2 VM

## [function] disassemble-waffe2-ir

```lisp
(disassemble-waffe2-ir toplevel &key (backward t) (stream t) (fuse-p t))
```

Prints out the compiled cl-waffe2 IR from toplevel to each leaf points to `stream`. If `backward` was set to t, `backward` is also displayed.

## [function] benchmark-accept-instructions

```lisp
(benchmark-accept-instructions iseq &key (n-sample 1) (ignore-first-call nil) (stream t) (top-k 10))
```

Basically, the function `benchmark-accept-instruction` executes the given list of instructions with profiling execution time, but at the end of proess, displays the report into `stream`.

## Inputs

`n-sample[fixnum]` repeats the iseq execution for `n-sample` times

`ignore-first-call[boolean]` If t, ignores the first call to avoid including allocating time.

`stream[stream]` the place to display the result

`top-k[fixnum]` top-k slowest nodes are displayed at the end of report.

## Return

`result[AbstractTensor]`

## Example

```lisp
CL-WAFFE2-REPL> (with-no-grad (benchmark-accept-instructions (compile-forward-and-backward (!softmax (randn `(128 128)))) :n-sample 1000))
 Time(s)   |   Instruction ( * - Beyonds the average execution time)
0.005366   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1078760 <= op(TID1078760(128 128) TID1078676(128 128))>
5.62e-4    | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID1078719 <= op(TID1078719(128 128) TID1078717(128 1))>
0.001171   | <WfInst[Compiled: SCALARMUL-CPUTENSOR]               : TID1078679 <= op(TID1078679(128 1) TID1078681(1))>
3.62e-4    | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID1078690 <= op(TID1078690(128 128) TID1078679(128 1))>
0.084953*  | <WfInst[Compiled: ADDNODE-CPUTENSOR]                 : TID1078690 <= op(TID1078690(128 128) TID1078676(128 128))>
0.053719*  | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1078719 <= op(TID1078719(128 128) TID1078690(128 128))>
6.69e-4    | <WfInst[Compiled: MOVESCALARTENSORNODE-SCALARTENSOR] : TID1078742 <= op(TID1078742(1) TID1078714(1))>
0.120082*  | <WfInst[Compiled: SCALARDIV-CPUTENSOR]               : TID1078719 <= op(TID1078719(128 128) TID1078742(1))>
0.049947*  | <WfInst[Compiled: SUBNODE-CPUTENSOR]                 : TID1078760 <= op(TID1078760(128 128) TID1078719(128 128))>
0.004672   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1078839 <= op(TID1078839(128 128) TID1078760(128 128))>
0.166431*  | <WfInst[Compiled: EXPNODE-LISPTENSOR]                : TID1078839 <= op(TID1078760(128 128) TID1078839(128 128))>
0.004736   | <WfInst[Compiled: <DELETED>]                         : TID1078856 <= op(TID1078856(128 128) TID1078839(128 128))>
0.001068   | <WfInst[Compiled: SCALARMUL-CPUTENSOR]               : TID1078805 <= op(TID1078805(128 1) TID1078807(1))>
4.96e-4    | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID1078816 <= op(TID1078816(128 128) TID1078805(128 1))>
0.004744   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1078788 <= op(TID1078788(128 128) TID1078760(128 128))>
0.165539*  | <WfInst[Compiled: EXPNODE-LISPTENSOR]                : TID1078788 <= op(TID1078760(128 128) TID1078788(128 128))>
0.085777*  | <WfInst[Compiled: ADDNODE-CPUTENSOR]                 : TID1078816 <= op(TID1078816(128 128) TID1078788(128 128))>
0.050755*  | <WfInst[Compiled: DIVNODE-CPUTENSOR]                 : TID1078856 <= op(TID1078856(128 128) TID1078816(128 128))>

18 Instructions | 15 Tensors

 Total Time: 0.801049 sec

 Instruction                                         | Total time (s) | Time/Total (n-sample=1000)
<WfInst[Compiled: EXPNODE-LISPTENSOR]                | 0.33196998   | 41.441906%
<WfInst[Compiled: ADDNODE-CPUTENSOR]                 | 0.17073      | 21.313303%
<WfInst[Compiled: SCALARDIV-CPUTENSOR]               | 0.120082     | 14.990594%
<WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          | 0.068501     | 8.551412%
<WfInst[Compiled: DIVNODE-CPUTENSOR]                 | 0.050755     | 6.3360667%
<WfInst[Compiled: SUBNODE-CPUTENSOR]                 | 0.049947     | 6.2351995%
<WfInst[Compiled: <DELETED>]                         | 0.004736     | 0.5912248%
<WfInst[Compiled: SCALARMUL-CPUTENSOR]               | 0.002239     | 0.2795085%
<WfInst[Compiled: VIEWTENSORNODE-T]                  | 0.0014199999 | 0.17726754%
<WfInst[Compiled: MOVESCALARTENSORNODE-SCALARTENSOR] | 6.69e-4      | 0.08351549%
{CPUTENSOR[float] :shape (128 128) :named ChainTMP1078855 
  ((0.0017760922 0.0030971088 0.017302852  ~ 0.012318904  6.049352e-4  0.0041618845)                    
   (0.012581187  0.0030174912 0.016748475  ~ 0.007076549  0.007030908  0.0017801385)   
                 ...
   (0.0036988985 0.0061271163 0.05046869   ~ 0.009297135  0.003441493  5.820294e-4)
   (0.045387346  0.004674337  0.0018589711 ~ 0.008918608  0.0024204857 0.00761818))
  :facet :input
  :requires-grad NIL
  :backward NIL}
CL-WAFFE2-REPL>
;; (The result may not be the latest)
```


## [function] compile-forward-and-backward

```lisp
(compile-forward-and-backward toplevel &key (need-backward t) (fuse-p t) (compile-mode :default))
```

Compiles into cl-waffe2 IR from topleve to each leaf points (detach-p=t or backward=null variables). set `fuse-p`=t to get additional optimization to the generated IR.

Tips: `disassemble-waffe2-ir` to display compiled Instruction Sequence.

## Return

`(values forward-iseq backward-iseq leaves[an list of AbstractTensor that appeared in the node] dout)`

## [function] accept-instructions

```lisp
(accept-instructions iseq)
```

Evaluates generated cl-waffe2 IR sequence.

`iseq[list]` an list of `WFInstruction`

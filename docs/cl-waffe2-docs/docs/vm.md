
# cl-waffe2 VM

## [function] disassemble-waffe2-ir

```lisp
(disassemble-waffe2-ir toplevel &key (backward t) (stream t) (fuse-p t))
```

Prints out the compiled cl-waffe2 IR from toplevel to each leaf points to `stream`. If `backward` was set to t, `backward` is also displayed.


## [function] compile-forward-and-backward

```lisp
(compile-forward-and-backward toplevel &key (need-backward t) (fuse-p t) (compile-mode :default))
```

Compiles into cl-waffe2 IR from topleve to each leaf points (detach-p=t or backward=null variables). set `fuse-p`=t to get additional optimization to the generated IR.

Tips: `disassemble-waffe2-ir` to display compiled Instruction Sequence.

## [function] accept-instructions

```lisp
(accept-instructions iseq)
```

Evaluates generated cl-waffe2 IR sequence.

`iseq[list]` an list of `WFInstruction`
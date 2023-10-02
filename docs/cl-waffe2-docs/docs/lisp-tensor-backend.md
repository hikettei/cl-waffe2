
# [backend] :cl-waffe2/backends.lisp

The package `:cl-waffe2/backends.lisp` provides a `LispTensor` backend which is designed with the aim of portalibity, not performance. All matrix operations are performed via kernels written in ANSI Common Lisp.

In order to use this backend, add this line:

```lisp
(with-devices (LispTensor)
   ;; body
   )
```

## [AbstractTensor] LispTensor

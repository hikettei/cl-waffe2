
# [package] :cl-waffe2/backends.lisp
The package `:cl-waffe2/backends.lisp` provides an AbstractTensor `LispTensor` as an external backend, and designed with the aim of portalibity, not performance. Therefore, most implementations of this follow ANSI Common Lisp, so it will work in any environment but concerns remain about speed.

It is recommended that `LispTensor` are installed in the lowest priority of `*using-backend*`, and `Couldnt find any implementation for ...` error will never occurs.
## [AbstractTensor] LispTensor

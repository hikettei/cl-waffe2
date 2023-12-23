
(cl:in-package :cl-user)

(defpackage :cl-waffe2/backends.metal
  (:documentation
   "## [package] :cl-waffe2/backends.metal
This package provides a Metal accelerated extension for cl-waffe2, including JIT Compiler.

```
$ qlot install

```
")
  (:use
   :cl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes)
  (:export
   #:MetalTensor
   ))


(in-package :cl-waffe2/backends.metal)


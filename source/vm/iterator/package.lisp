
(in-package :cl-user)

(defpackage :cl-waffe2/vm.iterator
  (:documentation
   "
## [package] :cl-waffe2/vm.iterator

This package provides data structures and functions for regular (i.e.: stride*index+offset) access to matrices
and their optimization. IRs (WfInstruction) and iteration-related matrix operations (e.g.: !permute, !view) depends
this package, for example:

- Access rules for array elements are computed with delayed evaluation by the structure `range`.
    - (!view x `(0 10 -2)) creates a view, view is comprised of various range.

- Abstraction without inconvenience: Split IRs from larger to small.
    - This package works as a more fundamental level, and it is dedicated to the runtime code generation.
    - Principle Operations (AbstractNodes) are first converterd to WfInstruction (Extended Wengert List)
        - If the operation is ongoing without the runtime code generation, WfInstruction is the smallest unit of operations.
        - With the runtime generation enabled, we need something more primitive and small structure, otherwise
          The dependency of array accessing considering parallelization would be discarded.
          - So we use `Polyhedral Model` to express the computation in the more fundamental level.
            And minimize the loss to find the best combinations.
            If one wants to implement JIT Compiler, they should consider implementing: Compiler from `Optimized Polyhedral IR` -> `Any Backends`.

### Usage

The basic workflow is following:

```
1. Actions + IndexRef as a High-Level IR

2. Creating a polyhedral IR

3. Optimizing

4. Returning an optimized version of the IR represented as Actions + IndexRef IR
```

")
  (:nicknames #:wf/iter)
  (:use
   :cl
   :cl-waffe2/vm
   :cl-waffe2/vm.generic-tensor
   :alexandria
   :linear-programming)
  (:export
   #:range
   #:range-size
   #:range-from
   #:range-start-index
   #:range-to
   #:range-step
   #:range-nth
   #:do-range
   #:.range)
  (:export
   ;; TODO: Export features related to scheduling
   )
  (:export
   #:trace-invocation
   #:solve-invocations
   ))

(in-package :cl-waffe2/vm.iterator)

(defun butnil (list) (loop for l in list if l collect l))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

(defun range-list (upfrom below &optional (by 1))
  (loop for i upfrom upfrom below below by by collect i))


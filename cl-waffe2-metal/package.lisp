
(cl:in-package :cl-user)

(defpackage :cl-waffe2/backends.metal
  (:documentation
   "## [package] :cl-waffe2/backends.metal

MPS Backend for cl-waffe2

# Installling cl-metal

```
$ qlot install
```
")
  (:use
   :cl
   :cl-metal
   :cl-waffe2/base-impl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes)
  (:export
   #:MetalTensor
   ))

(in-package :cl-waffe2/backends.metal)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find :mgetal *features*)
    (when (metal-available-p)
      (push :metal *features*)))

  (declaim (inline metal-reject-p))
  (defun metal-reject-p ()
    #+metal(progn t)
    #-metal(progn nil)))

;; If Metal is available on the environment, cl-waffe2 prefers to use Metal instead of CPUTensor.
#+metal(setf cl-waffe2/vm.generic-tensor:*using-backend* `(MetalTensor cl-waffe2/backends.lisp:LispTensor))


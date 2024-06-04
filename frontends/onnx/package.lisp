
(cl:defpackage :cl-waffe2.frontends/onnx
  (:documentation "
## [package] cl-waffe2.frontends/onnx

From ONNX to cl-waffe2 IR Converter.

### Usage

1. Add `cl-waffe2.frontends/onnx` in your asdf configuration.

```lisp
(asdf:defsystem ...
    ...
    :depends-on (\"cl-waffe2\" \"cl-waffe2.frontends/onnx\")
    ...)
```

2. Use the `from-model-proto` function to obtain the translated model.

```lisp
(cl-waffe2.frontends/onnx:from-model-proto
    (cl-onnx:load-model \"path_to_your_model.onnx\"))
```
")
  (:use :cl :cl-onnx)
  (:import-from
   :cl-waffe2/vm.generic-tensor
   :make-input
   :make-tensor
   :AbstractTensor)
  (:import-from
   :cl-waffe2
   :change-facet)
  (:export
    #:from-model-proto))

(cl:in-package :cl-waffe2.frontends/onnx)

;; Turn off visualization features for larger models.
(setf cl-onnx::*visualize* t)

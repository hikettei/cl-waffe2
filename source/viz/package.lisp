
(in-package :cl-user)

(defpackage :cl-waffe2/viz
  (:use :cl :cl-waffe2/vm.nodes :cl-waffe2/vm.generic-tensor :cl-waffe2/base-impl)
  (:export #:viz-computation-node #:dprint))

(in-package :cl-waffe2/viz)



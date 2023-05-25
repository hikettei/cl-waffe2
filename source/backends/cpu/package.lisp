
(in-package :cl-user)

(defpackage :cl-waffe2/backends.cpu
  (:use :cl :cl-waffe2/vm.generic-tensor :cl-waffe2/vm.nodes)
  (:export
   :CPUTensor))

(in-package :cl-waffe2/backends.cpu)

(setf cl-waffe2/vm.generic-tensor:*using-backend* `(cl-waffe2/backends.cpu:CPUTensor))



(in-package :cl-waffe2/vm.generic-tensor)

(defclass DebugTensor (AbstractTensor) nil) ;; ANSI CL's (make-array)

(defclass CPUTensor (AbstractTensor) nil) ;; SBCL's make-array

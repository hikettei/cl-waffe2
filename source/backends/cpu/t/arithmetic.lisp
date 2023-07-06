
(in-package :cl-waffe2/backends.cpu.test)

(in-suite :test-backends-cpu)

(eval-when (:compile-toplevel :load-toplevel :execute)
  
(add-tester CPUTensor)
(sub-tester CPUTensor)
(move-tester CPUTensor)

(matmul-test-set CPUTensor)

)


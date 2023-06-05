
(in-package :cl-waffe2/backends.cpu)

;; experiment
;; What if [10 10] + [10 1] speaking of memory-layout ????
;; define-vop
;; Before working with cpu-backend, dtypeのspecification

;; OpenBLAS Kernelは後回しにする・・・
;; make generics.


(define-impl (AddNode :device CPUTensor)
	     :forward ((self x y)

		       )
	     :backward ((self dout dx dy)

			))



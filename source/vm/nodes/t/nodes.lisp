
(in-package :cl-waffe2/vm.nodes.test)

(in-suite :test-nodes)

;; After Implementing Tensor

(defnode (Bijective-Function (myself)
	  :where `([x y] [x y] -> [x y])
	  :documentation "Bijective-Function has a one-to-one correspondence."))


(define-impl (Bijective-Function :device CPUTensor)
	     :forward ((self x)
		       `(values ,x))
	     :backward ((self dy)
			`(values ,dy)))

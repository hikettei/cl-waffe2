
(in-package :cl-waffe2/vm.nodes.test)

(in-suite :test-nodes)

;; After Implementing Tensor

(defnode (Bijective-Function (myself)
	  :where `([~] [~] -> [~])
	  :documentation "Bijective-Function has a one-to-one correspondence."))

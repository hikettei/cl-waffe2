
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


(defnode (AddNode (myself)
	  :where `([x y] [x y] -> [x y])
	  :documentation "x + y"))

(defnode (MulNode (myself)
	  :where `([x y] [x y] -> [x y])
	  :documentation "x * y"))

(defnode (TransposeNode (myself)
	  :where `([x y] -> [y x])
	  :documentation "x.T"))


(define-impl (AddNode :device CPUTensor)
	     :forward ((self x y)
		       `(+ ,x ,y))
	     :backward ((self dy)
			`(values ,dy ,dy)))

(define-impl (MulNode :device CPUTensor)
	     :forward ((self x y)
		       `(* ,x ,y))
	     :backward ((self dy)
			`(values ,dy ,dy)))

(defun build-test ()
  (let ((x (make-tensor `(10 10)))
	(y (make-tensor `(10 10)))
	(z (make-tensor `(10 10))))

    (forward (TransposeNode)
	     (forward (AddNode)
		      (forward (MulNode) x y) z))))



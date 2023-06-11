
(in-package :cl-waffe2/vm.nodes.test)

(in-suite :test-nodes)

;; After Implementing Tensor

(defnode (Bijective-Function (myself)
	  :where ([x y] [x y] -> [x y])
	  :documentation "Bijective-Function has a one-to-one correspondence."))

(defnode (Transpose-Function (myself)
	  :where ([x y] -> [y x])
	  :documentation "x y -> y x"))


(eval-when (:compile-toplevel :load-toplevel :execute)
  (defclass MyBackend (AbstractTensor) nil)
  (defclass MyBackend-With-Impl (AbstractTensor) nil))

(define-impl (Bijective-Function :device CPUTensor)
	     :forward ((self x y)
		       `(values ,x ,y))
	     :backward ((self dout dx dy)
			(declare (ignore dx dy))
			(values dout dout)))

(define-impl (Transpose-Function :device CPUTensor)
	     :forward ((self x)
		       `(values ,x))
	     :backward ((self dout dx)
			(declare (ignore dx))
			(values dout)))


(define-impl (Bijective-Function :device MyBackend-With-Impl)
	     :forward ((self x)
		       `(values ,x))
	     :backward ((self dout dy)
			(declare (ignore dy))
			(values dout)))

(defun test-switch-backend1 ()
  (with-devices (CPUTensor)
    (typep (Bijective-Function) 'CL-WAFFE2/VM.NODES.FACETS-TMP::BIJECTIVE-FUNCTION-CPUTENSOR)))

(defun test-switch-backend2 ()
  (with-devices (MyBackend CPUTensor)
    (typep (Bijective-Function) 'CL-WAFFE2/VM.NODES.FACETS-TMP::BIJECTIVE-FUNCTION-CPUTENSOR)))

(defun test-switch-backend3 ()
  (with-devices (MyBackend-with-impl CPUTensor)
    (typep (Bijective-Function) 'CL-WAFFE2/VM.NODES.FACETS-TMP::BIJECTIVE-FUNCTION-MYBACKEND-WITH-IMPL)))


(test heuristic-backend-test
  (is (test-switch-backend1))
  (is (test-switch-backend2))
  (is (test-switch-backend3)))

(defun shape-test ()
  (let ((out (forward (Transpose-Function)
		      (forward (Bijective-Function)
			       (make-tensor `(10 3))
			       (make-tensor `(10 3))))))
    (equal (shape out) `(3 10))))

(test shape-test
  (is (shape-test)))


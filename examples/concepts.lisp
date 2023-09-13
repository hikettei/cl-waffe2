
(defpackage :concepts
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/nn
   :cl-waffe2/distributions
   :cl-waffe2/base-impl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/vm
   :cl-waffe2/optimizers

   :cl-waffe2/backends.cpu))

(in-package :concepts)

(defnode (MatMulNode-Revisit (self)
	  :where (A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :backward ((self dout a b c)
		     (declare (ignore c))
		     (values
		      (!matmul dout (!t b))
		      (!matmul (!t a) dout)
		      nil))
	  :documentation "OUT <- GEMM(1.0, A, B, 0.0, C)"))

(defclass MyTensor (CPUTensor) nil)

;; This function should not be used for practice
;; Just example
(defun gemm! (m n k a-offset a b-offset b c-offset c)
  "Computes A[M N] @ B[N K] -> C[M K]"
  (declare (type (simple-array single-float (*)) a b c)
	   (type (unsigned-byte 32) m n k a-offset b-offset c-offset)
	   (optimize (speed 3)))
  (dotimes (mi m)
    (dotimes (ni n)
      (dotimes (ki k)
	(incf (aref c (+ c-offset (* mi K) ni))
	      (* (aref a (+ a-offset (* mi n) ki))
		 (aref b (+ b-offset (* ki k) ni))))))))

(define-impl (MatmulNode-Revisit :device MyTensor)
	     :save-for-backward (t t nil)
	     :forward ((self a b c)
		       `(,@(call-with-view
			    #'(lambda (a-view b-view c-view)
				`(gemm!
				  ,(size-of a-view 0)
				  ,(size-of b-view 0)
				  ,(size-of c-view 1)
				  ,(offset-of a-view 0) (tensor-vec ,a)
				  ,(offset-of b-view 0) (tensor-vec ,b)
				  ,(offset-of c-view 0) (tensor-vec ,c)))
			    `(,a ,b ,c)
			    :at-least-dim 2)
			 ,c)))

(defun test-gemm ()
  (with-devices (CPUTensor MyTensor)
    (let ((a (ax+b `(10 10) 1 0))
	  (b (ax+b `(10 10) 1 0))
	  (c (ax+b `(10 10) 0 0)))
      (proceed
       (call (MatmulNode-Revisit)
	     a b c)))))


(in-package :cl-waffe2/backends.cpu)

;; experiment
;; What if [10 10] + [10 1] speaking of memory-layout ????
;; define-vop
;; Before working with cpu-backend, dtypeのspecification

;; OpenBLAS Kernelは後回しにする・・・
;; make generics.

(defun add-matrix (x y offsetx offsety size)
  (declare (optimize (speed 3))
	   (type (simple-array single-float (*)) x y)
	   (type fixnum offsetx offsety size))
  (dotimes (i size)
    (incf (aref x (+ offsetx i)) (aref y (+ offsety i)))))


(define-impl (AddNode :device CPUTensor)
	     :forward ((self x y)
		       `(,@(call-with-view
			    #'(lambda (x-view
				       y-view)
				`(add-matrix
			 	  (tensor-vec ,x)
				  (tensor-vec ,y)
				  ,(offset-of x-view 0)
				  ,(offset-of y-view 0)
				  ,(size-of x-view 0)))
			    `(,x ,y))
			 ,x))
	     :backward ((self dout dx dy)
			(declare (ignore dx dy))
			(values dout dout)))



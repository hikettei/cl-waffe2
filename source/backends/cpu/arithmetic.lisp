
(in-package :cl-waffe2/backends.cpu)

;; experiment
;; What if [10 10] + [10 1]????

(defun add-matrix (x y offsetx offsety size)
  (declare (optimize (speed 3) (safety 0))
	   (type (simple-array single-float (*)) x y)
	   (type fixnum offsetx offsety size))
  (dotimes (i size)
    (incf (aref x (+ offsetx i)) (aref y (+ offsety i)))))



(define-impl (AddNode :device CPUTensor)
	     :forward ((self x y)
		       `(progn
			  ,(call-with-view
			    #'(lambda (x-view
				       y-view)
				`(add-matrix
				  (tensor-vec ,x)
				  (tensor-vec ,y)
				  ,(viewinstruction-offset x-view)
				  ,(viewinstruction-offset y-view)
				  ,(viewinstruction-size x-view)))
			    x y)
			  ,x))
	     :backward ((self dy)
			`(values ,dy ,dy)))


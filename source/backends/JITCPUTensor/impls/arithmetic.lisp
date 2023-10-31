
(in-package :cl-waffe2/backends.jit.cpu)

;; AbstractNodes which extends CPUJIT-Blueprint
;;  => Later JIT-Compiled by the event of on-finalizing-compiling.
;; It :forward slot gives no vaild ops
;; Instead, load-instructions gives an implementation that can be fused or deleted later.
(macrolet ((define-arith-impl (name fname)
	     `(progn
		(define-impl (,name
			      :device JITCPUTensor
			      :extends (CPUJIT-Blueprint))
			     :forward ((self x y)
				       (declare (ignore y))
				       ;; Its actual forward definition is described in load-instructions:
				       ;; Later compiled
				       `(progn ,x)))
		(defmethod load-instructions ((node ,name) &rest inputs)
		  (list
		   (make-inst :modify ,fname (car inputs) (cdr inputs)))))))
  (define-arith-impl AddNode "+=")
  (define-arith-impl SubNode "-=")
  (define-arith-impl MulNode "*=")
  (define-arith-impl DivNode "/="))

(define-impl (MoveTensorNode :device JITCPUTensor :extends (CPUJIT-Blueprint))
	     :forward ((self out target)
		       (declare (ignore target))
		       ;; Move: out <- target
		       `(progn ,out)))

(defmethod load-instructions ((node MoveTensorNode) &rest inputs)
  (list
   (make-inst :modify "=" (car inputs) (cdr inputs))))


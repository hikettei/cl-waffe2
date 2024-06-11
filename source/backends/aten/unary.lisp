
(in-package :cl-waffe2/backends.aten)

(macrolet ((impl (name op)
	     `(define-impl (,name :device Aten :extends (AtenOp))
			   :forward
			   ((self x out)
			    (prog1
				nil
			      (setf (aten-bp self) (unary->aten-code (list x out) 1 #'(lambda (x out) `((setf ,out (,',op ,x)))))
				    (aten-inputs self) (list x out)
				    (aten-outputs self) (list out)))))))
  (impl AbsNode abs)
  (impl SignNode sign)
  (impl SqrtNode sqrt)
  
  (impl SinNode sin)
  (impl CosNode cos)
  (impl TanNode tan)
  (impl ASinNode asin)
  (impl ACosNode acos)
  (impl ATanNode atan)
  (impl SinHNode sinh)
  (impl CosHNode cosh)
  (impl TanHNode tanh)

  (impl ASinHNode asinh)
  (impl ACosHNode acosh)
  (impl ATanHNode atanh)

  (impl ExpNode exp)
  (impl log2Node log2)
  (impl log10Node log10)
  (impl logeNode log)
  (impl Log1pNode log1p)
  )

(define-impl (ExptNode :device Aten :extends (AtenOp))
	     :forward ((self x out n)
		       (prog1
			   nil
			 (setf (aten-bp self) (unary->aten-code (list x out) 1 #'(lambda (x out) `((setf ,out (expt ,x ,(tensor-id n))))))
			       (aten-inputs self) (list x out n)
			       (aten-outputs self) (list out)))))

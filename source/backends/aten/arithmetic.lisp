
(in-package :cl-waffe2/backends.aten)

(macrolet ((impl (name op)
	     `(define-impl (,name :device Aten :extends (AtenOp))
			   :forward
			   ((self x y)
			    (prog1
				nil
			      (setf (aten-bp self) (unary->aten-code (list x y) 1 #'(lambda (x y) `((setf ,x (,',op ,x ,y)))))
				    (aten-inputs self) (list x y)
				    (aten-outputs self) (list x)))))))
  
  (impl AddNode +)
  (impl SubNode -)
  (impl MulNode *)
  (impl DivNode /)
  )


(define-impl (MoveTensorNode :device Aten :extends (AtenOp))
	     :forward
	     ((self x y)
	      (prog1
		  nil
		(setf (aten-bp self) (unary->aten-code (list x y) 1 #'(lambda (x y) `((setf ,x ,y))))
		      (aten-inputs self) (list x y)
		      (aten-outputs self) (list x)))))



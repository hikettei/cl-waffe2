
(in-package :cl-waffe2/backends.jit.cpu)

(macrolet ((define-arith-impl (name fname)
	     `(progn
		(define-impl (,name
			      :device JITCPUTensor
			      :extends (CPUJIT-Blueprint))
			     :forward ((self x y)
				       (let ((f (invoke-compiler
						 (symbol-name (gensym (symbol-name ',name)))
						 (list
						  (make-inst :modify ,fname x (list y))))))
					 `(progn
					    (funcall ,(jit-funcall-form f) ,@(jit-args f))
					    ,x)))))))
  (define-arith-impl AddNode "+=")
  (define-arith-impl SubNode "-=")
  (define-arith-impl MulNode "*=")
  (define-arith-impl DivNode "/="))


(define-impl (MoveTensorNode :device JITCPUTensor :extends (CPUJIT-Blueprint))
	     :forward ((self out target)
		       ;; Move: out <- target
		       (let ((f (invoke-compiler
				 (symbol-name (gensym "MOVE"))
				 (list
				  (make-inst :modify "=" out (list target))))))
			 `(progn
			    (funcall ,(jit-funcall-form f) ,@(jit-args f))
			    ,out))))


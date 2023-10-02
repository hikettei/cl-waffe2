
(in-package :cl-waffe2/backends.jit.cpu)

#|
(macrolet ((define-arith-impl (name lisp-op op-name)
	     `(progn
		(define-impl (,name
			      :device JITCPUTensor
			      :extends (CPUJIT-Blueprint))
			     :forward ((self x y)
				       ;; Called at a Toplevel
				       (progn
					 (setf (blueprint-use-var self) `(,x ,y))
					 (setf (blueprint-opecode self) ',lisp-op)
					 nil)

				       ;; Embedding into JIT
				       `(progn ,x)))

		(defmethod translate-op ((opcode (eql ',lisp-op)) opAST &rest args)
		  (make-inst :modify
			     ,op-name
			     (car args)
			     (cdr args))))))
  (define-arith-impl AddNode + "+=")
  (define-arith-impl SubNode - "-=")
(define-arith-impl MulNode * "*=")
(define-arith-impl DivNode / "/="))
|#

(define-impl (MoveTensorNode :device JITCPUTensor :extends (CPUJIT-Blueprint))
	     :forward ((self out target)
		       ;; Move: out <- target
		       (let ((f (invoke-compiler
				 (symbol-name (gensym "M"))
				 (list
				  (make-inst :modify "=" out (list target))))))
			 `(progn
			    (funcall ,(jit-funcall-form f) ,@(jit-args f))
			    ,out))))


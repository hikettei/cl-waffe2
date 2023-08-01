
(in-package :cl-waffe2/backends.jit.cpu)

;; ~~ TODO ~~~~~~~~~~~~~~~~~~~~~~
;;
;; optimize computation nodes
;; compose and fuse several operations
;; pruning unused computation nodes
;; the behaviour sometime wrong without with-no-grad
;; restrict option, disassemble it.



;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Generating a C Code from cl-waffe2.
;; The scope of JIT: Whether the generated code can be expressed with only one `for`.
;; Most of SIMD part relies on pragma simd.
;;

;; ===============================================================================
;;  Event Handlers
;; ===============================================================================

(defun apply-compile-p (variable next-variable)
  "Following the defition of 3., return t if there's a need to run compiling."

  ;; ViewTensorNode, PermuteTensorNode -> Compile -> ...
  ;;       ^ :device=t
  (declare (ignore variable))
  (or
   ;; If One of next variables are performed in different devices, or didn't exist in the first place (i.e.: is the end of nodes):
   (null next-variable)
   (not (typep next-variable 'JITAbleTensors))
   (not (typep (tensor-backward next-variable) 'CPUJIT-Blueprint))

   ;; JITCPUTensor do not provide nodes that change shapes of tensors
   ;; The change of shapes is detected:
   ;;(and
   ;; (not
   ;;  (cl-waffe2/vm.generic-tensor::shape-equal-list (print (shape variable)) (print (shape next-variable)))))

   ))

(defparameter *compiling-ntime-count* 0)
(defvar *in-place-routes* nil)

;; place <- actual-tensor
(defun register-in-place-mutation (place actual-tensor)
  (push (cons place actual-tensor) *in-place-routes*))

(defun int-sap-id (tensor)
  (symb (tensor-id tensor) '-sap))

;; Note: eval it when called with vm-build?
(defmethod on-finalizing-compiling ((current-node CPUJIT-Blueprint)
				    variable
				    next-variable)
  "If the node is needed to be compiled, compile."
  (if (apply-compile-p variable next-variable)
      (let ((*in-place-routes*)
	    (jit-function-name (symbol-name (gensym "CL_WAFFE2_C_KERNEL"))))
	(incf *compiling-ntime-count* 1)
	;;(format t "[INFO] Compiling nodes from ~a...~%" current-node)
	;; Pass these informations to invoke-compiler! function
        (multiple-value-bind (arguments tensors scalars source) (invoke-compiler! jit-function-name variable)
	  (load-foreign-function source)

	  (when *viz-compiled-code*
	    (format t "== [Log: JITCPUTensor] ===============~%~a~%" source))
	  (let ((call-form
		  (if (null tensors)
		      ;; -> arguments = Scalar
		      (expand-funcall-form
		       jit-function-name
		       arguments
		       nil)
		      ;; -> arguments = Scalar + Matrix or Matrix
		      (call-with-view
		       #'(lambda (&rest views)
			   (expand-funcall-form jit-function-name arguments views))
		       tensors
		       :at-least-dim 1))))
	    `(cffi:with-foreign-objects
		 (,@(loop for scal in scalars
			  collect `(,(int-sap-id scal) ,(dtype scal))))
	       (setf ,@(loop for scal in scalars
			     append `((cffi:mem-ref ,(int-sap-id scal) ,(dtype scal)) (tensor-vec (read-result ,scal)))))
	       ,call-form
	       
	       ;; Synchronize ScalarTensors
	       (setf ,@(loop for scal in scalars
			     append `((tensor-vec (read-result ,scal)) (cffi:mem-ref ,(int-sap-id scal) ,(dtype scal)))))

	       ;; (!sin x) isn't working while (!copy (!sin x)) is ok.
	       (print ,variable)
	       (print (read-result ,variable))
	       
	       ;; Synchronize In-place (for scalar tensor?)
	       (let* (,@(loop for case in (reverse *in-place-routes*)
			      ;; Sort by tensor-id
			      collect `(,(tensor-id (car case)) (read-result ,(car case)))))
		 (setf ,@(loop for case in (reverse *in-place-routes*)
			       append `((tensor-vec ,(tensor-id (car case)))
					(tensor-vec (read-result ,(cdr case)))))))))))
      nil))



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


;; apply-compile-p:
;;
;; A(3, 3)   ----------\
;;                     | --- out
;; B(3, 1*3) --[COPY] -/      ^apply-compile-p=t
;;               ^apply-compile-p=t
;;
;; detected one of: end of nodes, the argument is broadcasted, the change of devices

;;
;; variable \
;;           next-variable 
;; variable /
;;

(defun apply-compile-p (variable next-variable)
  "Following the defition of 3., return t if there's a need to run compiling."

  ;; ViewTensorNode, PermuteTensorNode -> Compile -> ...
  ;;       ^ :device=t
  (declare (ignore variable))
  (or
   ;; If One of next variables are performed in different devices, or didn't exist in the first place (i.e.: is the end of nodes):
   (null next-variable)
   ;;(not (typep next-variable 'JITAbleTensors))
   (not (typep (tensor-backward next-variable) 'CPUJIT-Blueprint))
   ;;(not (eql (tensor-projected-p variable) (tensor-projected-p next-variable)))
   
   ;; Composing element-wise operations with the same iteration.
   ;; Split iteraton:
   (some #'tensor-projected-p (tensor-variables next-variable))
   ))

(defparameter *compiling-ntime-count* 0)
(defvar *in-place-routes* nil)

;; place <- actual-tensor
(defun register-in-place-mutation (place actual-tensor)
  (push (cons place actual-tensor) *in-place-routes*))

(defun int-sap-id (tensor)
  (symb (tensor-id tensor) '-sap))


(defparameter *caching-c-source* nil)
(defun maybe-load-foreign-function (source end-of-node-p)
  (setf *caching-c-source* (concatenate 'string
					(or *caching-c-source* "")
					(format nil "~%")
					source))
  
  (when (and end-of-node-p *caching-c-source*)
    (load-foreign-function *caching-c-source*)
    (setf *caching-c-source* nil)))

;; Note: eval it when called with vm-build?
(defmethod on-finalizing-compiling ((current-node CPUJIT-Blueprint)
				    variable
				    next-variable
				    compile-me)
  "If the node is needed to be compiled, compile."
  
  (if (apply-compile-p variable next-variable)
      (let ((*in-place-routes*)
	    (jit-function-name (symbol-name (gensym "CL_WAFFE2_C_KERNEL"))))
	(incf *compiling-ntime-count* 1)
	;;(format t "[INFO] Compiling nodes from ~a...~%" current-node)
	;; Pass these informations to invoke-compiler! function
        (multiple-value-bind (arguments tensors scalars source) (invoke-compiler! jit-function-name variable)

	  ;; [TODO]: multiple call of gcc may result low performance
	  ;; Cache it
	  (maybe-load-foreign-function source (or compile-me (null next-variable)))
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
	       
	       ;; Synchronize In-place
	       (let* (,@(loop for case in (reverse *in-place-routes*)
			      ;; Sort by tensor-id
			      collect `(,(tensor-id (car case)) (read-result ,(car case)))))
		 (setf ,@(loop for case in (reverse *in-place-routes*)
			       append `((tensor-vec ,(tensor-id (car case)))
					(tensor-vec (read-result ,(cdr case)))))))

	       
	       ;; [Bug] (proceed (!sin x)) isn't working while (proceed (!copy (!sin x))) is ok.
	       ;; Synchronize output if the last node is in-place
	       ,(let* ((all-tensors `(,@scalars ,@tensors))
		       (latest-result (find (tensor-id variable) all-tensors :test #'eql :key #'tensor-id)))
		  (when latest-result
		    `(setf (tensor-vec (read-result ,variable)) (tensor-vec (read-result ,latest-result)))))))))
      nil))


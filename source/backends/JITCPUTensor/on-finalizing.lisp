
(in-package :cl-waffe2/backends.jit.cpu)

;; This is a JIT Compiler from cl-waffe2 to C which is aimed to optimize operations whose call-with-view can't optimize:
;;  Subjects to be optimized: MoveTensorNode
;;   - 1. 与えられた引数でsolve-loop-orderする->遅いならreject-p nil
;;   - 2. solve-loop-orderの結果をもとにしてhogehoge
;;   - 3. JIT Compile and cache the result (<- Cache dekiru youni suru)

(defparameter *compiling-ntime-count* 0)
(defvar *in-place-routes* nil)

;; place <- actual-tensor
(defun register-in-place-mutation (place actual-tensor)
  (push (cons place actual-tensor) *in-place-routes*))

(defun int-sap-id (tensor)
  (symb (tensor-id tensor) '-sap))


(defparameter *caching-c-source* nil)
(defun maybe-load-foreign-function (source end-of-node-p)
  (when source
    (setf *caching-c-source* (concatenate 'string
					  (or *caching-c-source* "")
					  (format nil "~%")
					  source)))
  
  (when (and end-of-node-p *caching-c-source*)
    (load-foreign-function *caching-c-source*)
    (setf *seen* nil)
    (setf *caching-c-source* nil)))

;; Note: eval it when called with vm-build?
(defmethod on-finalizing-compiling ((current-node CPUJIT-Blueprint)
				    variable
				    next-variable
				    compile-me)
  "If the node is needed to be compiled, compile."
  (declare (ignore next-variable))

  (when compile-me
    (incf *compiling-ntime-count* 1)
    (let ((jit-fname (symbol-name (gensym "CL_WAFFE2_C_KERNEL"))))
      (invoke-compiler! jit-fname variable)
      )))
  
  (if (apply-compile-p variable next-variable)
      (let ((*in-place-routes*)
	    (jit-function-name (symbol-name (gensym "CL_WAFFE2_C_KERNEL"))))
	
	;;(format t "[INFO] Compiling nodes from ~a...~%" current-node)
	;; Pass these informations to invoke-compiler! function
        (multiple-value-bind (arguments tensors scalars source) (invoke-compiler! jit-function-name variable)

	  ;; Cache it
	  (maybe-load-foreign-function source compile-me)
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
	       (with-tensor-ptrs (,@(loop for tensor in arguments
					  if (typep tensor 'JITCPUTensor)
					    collect `(,(cPointer tensor) (read-result ,tensor))))
		 (locally (declare (optimize (speed 1)))
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
			`(setf (tensor-vec (read-result ,variable)) (tensor-vec (read-result ,latest-result)))))
		   (read-result ,variable)))))))
      nil))

(defmethod on-finished-compiling ((current-node (eql 'JITCPUTensor)))
  (maybe-load-foreign-function nil t))

(defmethod on-finished-compiling ((current-node (eql 'JITCPUScalarTensor)))
  (maybe-load-foreign-function nil t))


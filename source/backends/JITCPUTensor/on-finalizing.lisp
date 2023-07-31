
(in-package :cl-waffe2/backends.jit.cpu)

;; TODO:
;;
;; LispTensorより遅い場合がある！！！->計算ノード最適化！！
;; バグ修正：結果が0になって反映されてない
;; ノードの終端と途中で呼ばれる場合で色々違う

;; 演算の合成と計算ノードの最適化
;; restrict option disassemble it.



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
  
  (or
   ;; If One of next variables are performed in different devices, or didn't exist in the first place (i.e.: is the end of nodes):
   (null next-variable)
   (not (typep next-variable 'JITAbleTensors))
   (not (or (typep (tensor-backward next-variable) 'CPUJIT-Blueprint)
	    (typep (tensor-backward next-variable) 'CPUJIT-Scalar-Blueprint)))

   ;; The change of shapes is detected:
   (and
    (not
     (cl-waffe2/vm.generic-tensor::shape-equal-list (shape variable) (shape next-variable))))))

(defparameter *compiling-ntime-count* 0)

(defun int-sap-id (tensor)
  (symb (tensor-id tensor) '-sap))

;; Note: eval it when called with vm-build?
(defmethod on-finalizing-compiling ((current-node CPUJIT-Blueprint)
				    variable
				    next-variable)
  "If the node is needed to be compiled, compile."
  (if (apply-compile-p variable next-variable)
      (let ((jit-function-name (symbol-name (gensym "CL_WAFFE2_C_KERNEL"))))
	(incf *compiling-ntime-count* 1)
	;;(format t "[INFO] Compiling nodes from ~a...~%" current-node)
	;; Pass these informations to invoke-compiler! function
        (multiple-value-bind (arguments tensors scalars source) (invoke-compiler! jit-function-name variable)
	  (load-foreign-function source)
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
			     append `((cffi:mem-ref ,(int-sap-id scal) ,(dtype scal)) (tensor-vec ,scal))))
	       ,call-form
	       
	       (setf ,@(loop for scal in scalars
			     append `((tensor-vec ,scal) (cffi:mem-ref ,(int-sap-id scal) ,(dtype scal)))))
	       
	       (setf (cl-waffe2/vm.generic-tensor::statecontainer-forward-result (tensor-state ,variable))
		     (list ,variable))))))
      nil))

;; TO ADD:
;; compile option, avx2


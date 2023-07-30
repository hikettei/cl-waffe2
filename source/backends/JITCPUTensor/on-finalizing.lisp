
(in-package :cl-waffe2/backends.jit.cpu)

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Generating a C Code from cl-waffe2.
;; The scope of JIT: Whether the generated code can be expressed with only one `for`.
;; 
;; 



;; Memo: OffsetはLisp側で加算することにする
;; Pragma SIMDで自動ベクトル化
;; gemmはOpenBLASのAPIを呼び出す
;; void NAME ...

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


;; MEMO: proceedモードで動作するときはevalする？
(defmethod on-finalizing-compiling ((current-node CPUJIT-Blueprint)
				    variable
				    next-variable)
  "If the node is needed to be compiled, compile."
  (if (apply-compile-p variable next-variable)
      (progn
	(incf *compiling-ntime-count* 1)
	;;(format t "[INFO] Compiling nodes from ~a...~%" current-node)
	;; Pass these informations to invoke-compiler! function
        (invoke-compiler! "TMP_FUNCTION" variable)
	
	;; flowchart:
	;; call gcc
	;; return: `call-c-form
	
	)
      nil))

;; (defun call-c-form (tensors))


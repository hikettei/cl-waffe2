
(in-package :cl-waffe2/backends.jit.cpu)

;; Env in my macbook:
;; (cpujit-set-config
;;    :compiler "/usr/local/bin/gcc-13"
;;    :viz-compiled-code nil
;;    :openmp t)

(defparameter *lazy-c-source* "
(gensym).c
#pragma simd
<<Headers>>

<<Definitions>>
")

(defparameter *compiling-ntime-count* 0)

;; [Fix] 何回も同じコードをコンパイルするの？
;; [TODO] Im2Col Col2Im Fusion
;; [TODO] OpenMP

;; Workloads:
;;  - [Fix] 何回も同じコードをコンパイルするのか？
;;  - [Opt] Im2Col Col2Im
;;  - [Add] OpenMP (thresholds, nested)
;;  - [Add]

;; This is a toplevel of JIT-Compiling Backend
;; After High-Level IR compiling is finished, the method on-finalizing-compiling will be invoked.
;; Depending on *using-backend*, the method below would be also invoked.
(defmethod on-finalizing-compiling ((device-name (eql 'JITCPUTensor)) iseq-fw iseq-bw)
  (flet ((applying-jit-helper (iseq &aux (out nil))
	   ;; Replacing the op slot whose backend is JITxxxTensor
	   ;; with compiled C kernel
	   ;; without modifying orders of instructions
	   ;; If any, fuse several instructions
	   ;; To Eliminate Unused MoveTensorNode.

	   ;; WfInstruction: out-to <- op(args)
	   ;; args = Tensor | sv4bw(Tensor) ...
	   (loop for inst in iseq do
	     (typecase (wfop-node inst)
	       (CPUJIT-Blueprint
		;; [TODO] Stack as for element-wise operations
		(let ((ir (apply #'load-instructions (wfop-node inst) (tensor-variables (wfop-self inst)))))
		  ;; As of this writing, this backend does not provide features for FusionOPs
		  ;; So just replacing op is ok and wfop-args cause no conflicts
		  ;; But If the instruction is created by fusion several ops
		  ;; we have to note that create a new WfINstruction

		  ;; [TODO]
		  ;; compilers should be reluctant to insert a new c line which is not worth it.		  
		  (setf (wfop-op inst) (make-jit-compiled-op (symbol-name (gensym)) ir))
		  (push inst out)))
	       (T
		(push inst out))))
	   (reverse out)))

    ;; Resets buffers
    (let ((*lazy-c-source* ""))
      (multiple-value-bind (iseq-fw iseq-bw)
	  (values
	   (applying-jit-helper iseq-fw)
	   (applying-jit-helper iseq-bw))
	
	;; Compiles *lazy-c-source*
	;; And loads via CFFI as a shared library
	(when (not (string= *lazy-c-source* ""))
	  (when *viz-compiled-code*
	    (print *lazy-c-source*))
	  (load-foreign-function *lazy-c-source*))
	(values iseq-fw iseq-bw)))))


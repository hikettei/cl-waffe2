
(in-package :cl-waffe2/vm)

;; TODO:
;;  - 1. SaveForBackwardをIRに混ぜる 
;;  - 2. load-pointerを除外する (Moveにする？)
;;  - 3. acceptor.lisp からinvoke-jit-compiler

(defun unroll-save-for-backward (iseq)
  (declare (type list iseq))
  (flet ((ensure-no-sv4bw (iseq)
	   (loop for inst in iseq do	     
	     (setf (wfop-sv4bw inst) (make-list (length (wfop-args inst)))))
	   iseq))
    (ensure-no-sv4bw
     (butnil
      (alexandria:flatten
       (loop for inst in iseq
	     append
	     (append
	      (loop for sv4bw in (wfop-sv4bw inst)
		    for arg   in (wfop-args  inst)
		    if sv4bw
		      collect
		      (prog2
			  (setf (detach-p arg) t)
			  (node-compile-into-vm
			   (forward
			    (cl-waffe2/base-impl:MoveTensorNode (dtype arg))
			    ;; set A <- B
			    sv4bw
			    arg))
			(setf (detach-p arg) nil)))
	      (list inst))))))))

(defun invoke-jit-compiler (iseq
			    &aux
			      (iseq (unroll-save-for-backward iseq))
			      )
  (declare (type list iseq))

  (let ((invocations-stored)
	(jit-compiled-irs))

    (loop for instruction in iseq do
      (let ((invocations (get-invocations-from-node (wfop-node instruction) (wfop-args instruction))))
	(if invocations
	    (setq invocations-stored `(,@invocations-stored ,@invocations))
	    (progn
	      ;; Apply a jit compiling
	      (print (time (wf/iter:solve-invocations invocations-stored)))
	      
	      ;; Push a new WfInst
	      (setq invocations-stored nil)))))

    ;; The result is cached by the status of dynamic-shape at that time
    jit-compiled-irs
    iseq))



(in-package :cl-waffe2/vm)

(defun unroll-save-for-backward (iseq)
  "Unrolls SV4BW into a sequence of instructions"
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

(defun unroll-allocation (iseq)
  (let* ((allocations)
	 (out
	   (loop for inst in iseq
		 if (eql (wfop-op inst) #'lazy-clone-wreckage)
		   do (push inst allocations)
		 else
		   collect inst)))
    `(,@allocations
      ,@out)))

;; TODO: Adjustable Shapeが決定されていないとコンパイルが走らない
;; acceptor.lispレベルで管理する <The Shapes> -> Compiled
(defun invoke-jit-compiler (iseq
			    &aux
			      ;; Unrolls save for backward in order to fuse them
			      (iseq (unroll-save-for-backward iseq))
			      (iseq (unroll-allocation        iseq)))
  (declare (type list iseq))  
  (let ((invocations-stored)
	(result)
	(loadp-table (make-hash-table))
	(cached-jit-kernel (make-hash-table :test #'eql)))
    (labels ((read-tensor (tensor)
	       (or
		(gethash (tensor-id tensor) loadp-table)
		tensor))
	     (produce (schedule)
	       ;; Produces a fused and jit-compiled Lambda expressions
	       (or
		(gethash (wf/iter:scheduler-name schedule) cached-jit-kernel)
		(let ((out-to (reverse
			       (loop for (io . tensor) in (wf/iter:scheduler-args schedule)
				     if (or (eql io :io) (eql io :out))
				       collect (read-tensor tensor)))))
		  (setf
		   (gethash (wf/iter:scheduler-name schedule) cached-jit-kernel)
		   (make-wfop
		    (let ((f (wf/iter:schedule-codegen (car *using-backend*) schedule)))
		      #'(lambda (&rest args)
			  (apply f args)
			  (apply #'values out-to)))
		    (cdr (car (wf/iter:scheduler-args schedule)))
		    #'(lambda () (wf/iter:scheduler-name schedule))
		    (map 'list (compose #'read-tensor #'cdr) (wf/iter:scheduler-args schedule))
		    :out-to out-to))))))
      (loop for instruction in iseq do
	(if (wfop-loadp instruction)
	    (multiple-value-bind (x y) (read-loadp instruction)
	      (declare (ignore y))
	      ;; funcall ? ignoring ?
	      ;; TODO: How to deal with reshape/permute/view
	      (setf
	       (gethash (tensor-id x) loadp-table)
	       (apply
		(wfop-op instruction)
		(wfop-args instruction))))
	    (progn
	      (macrolet ((update (place)
			   `(setf ,place (map 'list #'read-tensor ,place))))
		(update (wfop-args instruction))
		(update (wfop-out-to instruction)))
	      (let ((invocations
		      (when (typep (wfop-node instruction) 'AbstractNode)
			(get-invocations-from-node
			 (wfop-node instruction)
			 (wfop-args instruction)))))
		(if invocations
		    (setq
		     invocations-stored
		     `(,@invocations-stored ,invocations))
		    (let ((schedules (wf/iter:solve-actions invocations-stored)))
		      (setf result `(,@result ,@(map 'list #'produce schedules) ,instruction))
		      ;; Push a new WfInst
		      (setq invocations-stored nil)))))))
      
      (let ((schedules (wf/iter:solve-actions invocations-stored)))	  
	(setf result `(,@result ,@(map 'list #'produce schedules))))
      ;; The result is cached by the status of dynamic-shape at that time
      result)))


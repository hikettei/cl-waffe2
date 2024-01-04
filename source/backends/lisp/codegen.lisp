
(in-package :cl-waffe2/backends.lisp)

;; This is the all of JIT Compiler from cl-waffe2 to Lisp.

;; TODO: lParallel
;; TODO: Disassemble
(defmethod cl-waffe2/vm.iterator:schedule-config ((backend-indicator (eql 'LispTensor)))
  ;; SIMD_Stride N_Threads
  (values
   nil
   4))

(defun parse-aref (ispace)
  `(aref ,(tensor-id (wf/iter:ispace-tensor ispace))
	 ,@(loop for dim in (wf/iter:ispace-space ispace)
		 collect
		 `(+
		   ,(wf/iter:iref-offset dim)
		   (the (signed-byte 32)
			(*
			 (the (signed-byte 32) ,(wf/iter:iref-stride dim))
			 (the (signed-byte 32) ,(wf/iter:iref-index dim))))))))

(defun render-op (action)
  (declare (type wf/iter:Action action))
  (let ((opname (wf/iter:action-op action))
	(sources (map 'list #'parse-aref (wf/iter:action-source action)))
	(targets (map 'list #'parse-aref (wf/iter:action-target action))))
    #.`(case opname
	 ,@(loop with arithmetic = `((AddNode . +)
				     (SubNode . -)
				     (MulNode . *)
				     (DivNode . /))
		 for (opname . val) in arithmetic
		 collect
		 `(,opname `(setf ,@targets (,',val ,@targets ,@sources))))
	 (MoveTensorNode `(setf ,@targets ,@sources))
	 ,@(loop with mathematical = `((SinNode . sin)
				       (CosNode . cos)
				       (TanNode . tan)
				       (ExpNode . exp))
		 for (opname . val) in mathematical
		 collect
		 `(,opname `(setf ,@targets (,',val ,@sources))))
	 (T
	  ;; TODO: make it error
	  (format t "No rewriting rule for ~a." opname)))))

(defmethod cl-waffe2/vm.iterator:schedule-codegen
    ((backend-indicator (eql 'LispTensor)) scheduler)
  (let ((source
	  `(lambda (,@(loop for (io . tensor) in (wf/iter:scheduler-args scheduler)
			    collect (tensor-id tensor)))
	     (declare
	      (optimize (speed 3) (safety 0))
	      ,(loop for (io . tensor) in (wf/iter:scheduler-args scheduler)
		     collect `(type (simple-array ,(dtype->lisp-type (dtype tensor)) (*)) ,(tensor-id tensor))))
	     
	     ,@(labels ((expand-helper (rest-stages)
			  (when (not (null (car rest-stages)))
			    (loop for stage in (car rest-stages)
				  collect
				  `(dotimes (,(wf/iter:iterstage-determines stage)
					     ,(wf/iter:iterstage-size stage))
				     ,@(loop for op in (wf/iter:iterstage-ops stage)
					     collect
					     (render-op op))
				     ,@(expand-helper (cdr rest-stages)))))))	 
		 (expand-helper (wf/iter:sort-stage scheduler))))))
    ;;(disassemble (compile nil source))
    source))


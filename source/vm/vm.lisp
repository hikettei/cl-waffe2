
(in-package :cl-waffe2/vm)

(defparameter *safety-mode-p* NIL "
## [parameter] *safety-mode-p*

When set to T, a run-time error is detected and a warning is displayed.")

(defparameter *logging-vm-execution* NIl "
## [parameter] *logging-vm-execution*

If set to T, the result is displayed on the terminal with the arguments used each time cl-waffe2 VM executes an instruction. In default, set to nil
")

(declaim (inline maybe-read-result write-result apply-instruction apply-inst-sv4bw))

(defmodel (SV4BW-Copier (self)
	   :on-call-> ((self x y)
		       (declare (ignore self))
		       (with-no-grad
			 (cl-waffe2/base-impl:!move x y :force t)))))

(defmodel-as (SV4BW-Copier)
  :where (A[~] B[~] -> OUT[~])
  :asif :function :named %vm-move)

(declaim (ftype (function (WfInstruction) t) apply-inst-sv4bw))
(defun apply-inst-sv4bw (instruction)
  (declare (type WfInstruction instruction)
	   (optimize (speed 3)))
  (when (not *no-grad*)
    (let ((variables (map 'list #'maybe-read-result (wfop-args instruction)))
	  (places    (wfop-sv4bw instruction)))
      (loop for var   in variables
	    for place in places
	    if (and place var) do
	      ;; Place <- Var
	      (%vm-move place var))))
  nil)

(declaim (ftype (function (AbstractTensor) AbstractTensor) maybe-read-result))
(defun maybe-read-result (tensor)
  (declare (type AbstractTensor tensor))
  (if (tensor-tmp-p tensor)
      tensor
      (if (scalar-p tensor)
	  tensor
	  (let* ((state (tensor-state tensor))
		 (res
		   (or (when state
			 (cl-waffe2/vm.generic-tensor::statecontainer-forward-result state))
		       tensor)))
	    (the AbstractTensor res)))))

(declaim (ftype (function (list list) t) write-result))
(defun write-result (tensors results)
  ;; [TODO] Runtime Shape-Error Detection
  (loop for tensor of-type AbstractTensor in tensors
	for result in results
	if  result do
	  (if (tensor-tmp-p tensor)
	      (progn
		;; Update Memory-Pool
		(setf (tensor-vec (read-from-mempool-tensor tensor)) (cl-waffe2/vm.generic-tensor::vec result))
		(tensor-vec result)
		(cl-waffe2/vm.generic-tensor::embody-tensor-vec tensor result)
		) ;; Tensor.ID <- Result
	      (if (scalar-p tensor)
		  (setf (tensor-vec tensor) (tensor-vec result))
		  (let* ((state (tensor-state tensor)))
		    (setf (cl-waffe2/vm.generic-tensor::statecontainer-forward-result state) result))))))

(declaim (ftype (function (WFInstruction) list) apply-instruction))
(defun apply-instruction (instruction)
  (declare (type WFInstruction instruction)
	   (optimize (speed 3)))
  
  (when *logging-vm-execution*
    (let* ((inst (format nil "~a" instruction))
	   (cnt  (length inst)))
      (format t "= [*logging-vm-execution*] ~a
Instruction: ~a
args:
~a"
	      (with-output-to-string (out)
		(dotimes (i cnt) (princ "=" out)))
	      inst
	      (map 'list #'maybe-read-result (wfop-args instruction)))))

  
  (let ((outs (multiple-value-list
	       (apply
		(the function (wfop-op instruction))
		(map 'list #'maybe-read-result (wfop-args instruction))))))

    
    (when *safety-mode-p*
      (when (some (the function (compose #'null #'cl-waffe2/vm.generic-tensor::vec)) outs)
	(warn "cl-waffe2 VM: Runtime Warning
The instruction: ~a

returned a tensor whose its storage vec is null. In runtime, cl-waffe2 VM excepts all returned tensors have a valid storages and the next arguments are overwritten with this invaild tensor. So this could be lead to Assertion Failed ... Error. If you believe this alert is false and desire to delete this warning, set *safety-mode-p*=nil.

out-to returned:

~a" instruction outs))

      (when (not (= (length outs) (length (wfop-out-to instruction))))
	(warn "cl-waffe2 VM: Runtime Warning
The instruction: ~a
should be return ~R arguments, but got ~R.

out-to returned:

~a"
	      instruction
	      (length outs)
	      (length (wfop-out-to instruction))
	      outs))

      (mapc #'(lambda (excepted received)
		(when (not (eql (the boolean (scalar-p excepted)) (scalar-p received)))
		  (warn "cl-waffe2 VM: Runtime Warning
The instruction: ~a
Scalars and Matrices are incompatible:
Excepted:
~a
Butgot:
~a"
			instruction
			excepted
			received))

		(when (not (equal (shape excepted) (shape received)))
		  (warn "cl-waffe2 VM: Runtime Warning
The instruction: ~a
Shapes are incompatible.
Excepted:
~a
Butgot:
~a"
			instruction
			excepted
			received)))
	    (wfop-out-to instruction) outs))

    (when *logging-vm-execution*
      (format t "
outs:
~a~%"
	      outs))
    outs))

(declaim (ftype (function (list) t) accept-instructions))
(defun accept-instructions (iseq)
  "
## [function] accept-instructions

```lisp
(accept-instructions iseq)
```

Evaluates generated cl-waffe2 IR sequence.

`iseq[list]` an list of `WFInstruction`
"
  (declare (type list iseq)
	   (optimize (speed 3)))

  (when iseq
    (loop for inst of-type WFInstruction in iseq
	  ;; TODO: Runtime Shape Inspection etc...
	  do (apply-inst-sv4bw inst)
	     (write-result (wfop-out-to inst) (apply-instruction inst))
	  finally
	     (return-from accept-instructions (apply #'values (map 'list #'maybe-read-result (wfop-out-to inst)))))))

(defparameter *under-benchmark-set* nil "(list sorted-node profiled-table) If there's any") 

;; TODO: Measure memory-usage
;; TODO: Include this to the documents
(defun benchmark-accept-instructions (iseq
				      &key
					(n-sample 1)
					(ignore-first-call nil)
					(stream t)
					(top-k 10))
  "
## [function] benchmark-accept-instructions

```lisp
(benchmark-accept-instructions iseq &key (n-sample 1) (ignore-first-call nil) (stream t) (top-k 10))
```

Basically, the function `benchmark-accept-instruction` executes the given list of instructions with profiling execution time, but at the end of proess, displays the report into `stream`.

## Inputs

`n-sample[fixnum]` repeats the iseq execution for `n-sample` times

`ignore-first-call[boolean]` If t, ignores the first call to avoid including allocating time.

`stream[stream]` the place to display the result

`top-k[fixnum]` top-k slowest nodes are displayed at the end of report.

## Return

`result[AbstractTensor]`

## Example

```lisp
CL-WAFFE2-REPL> (with-no-grad (benchmark-accept-instructions (compile-forward-and-backward (!softmax (randn `(128 128)))) :n-sample 1000))
 Time(s)   |   Instruction ( * - Beyonds the average execution time)
0.005366   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1078760 <= op(TID1078760(128 128) TID1078676(128 128))>
5.62e-4    | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID1078719 <= op(TID1078719(128 128) TID1078717(128 1))>
0.001171   | <WfInst[Compiled: SCALARMUL-CPUTENSOR]               : TID1078679 <= op(TID1078679(128 1) TID1078681(1))>
3.62e-4    | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID1078690 <= op(TID1078690(128 128) TID1078679(128 1))>
0.084953*  | <WfInst[Compiled: ADDNODE-CPUTENSOR]                 : TID1078690 <= op(TID1078690(128 128) TID1078676(128 128))>
0.053719*  | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1078719 <= op(TID1078719(128 128) TID1078690(128 128))>
6.69e-4    | <WfInst[Compiled: MOVESCALARTENSORNODE-SCALARTENSOR] : TID1078742 <= op(TID1078742(1) TID1078714(1))>
0.120082*  | <WfInst[Compiled: SCALARDIV-CPUTENSOR]               : TID1078719 <= op(TID1078719(128 128) TID1078742(1))>
0.049947*  | <WfInst[Compiled: SUBNODE-CPUTENSOR]                 : TID1078760 <= op(TID1078760(128 128) TID1078719(128 128))>
0.004672   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1078839 <= op(TID1078839(128 128) TID1078760(128 128))>
0.166431*  | <WfInst[Compiled: EXPNODE-LISPTENSOR]                : TID1078839 <= op(TID1078760(128 128) TID1078839(128 128))>
0.004736   | <WfInst[Compiled: <DELETED>]                         : TID1078856 <= op(TID1078856(128 128) TID1078839(128 128))>
0.001068   | <WfInst[Compiled: SCALARMUL-CPUTENSOR]               : TID1078805 <= op(TID1078805(128 1) TID1078807(1))>
4.96e-4    | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID1078816 <= op(TID1078816(128 128) TID1078805(128 1))>
0.004744   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1078788 <= op(TID1078788(128 128) TID1078760(128 128))>
0.165539*  | <WfInst[Compiled: EXPNODE-LISPTENSOR]                : TID1078788 <= op(TID1078760(128 128) TID1078788(128 128))>
0.085777*  | <WfInst[Compiled: ADDNODE-CPUTENSOR]                 : TID1078816 <= op(TID1078816(128 128) TID1078788(128 128))>
0.050755*  | <WfInst[Compiled: DIVNODE-CPUTENSOR]                 : TID1078856 <= op(TID1078856(128 128) TID1078816(128 128))>

18 Instructions | 15 Tensors

 Total Time: 0.801049 sec

 Instruction                                         | Total time (s) | Time/Total (n-sample=1000)
<WfInst[Compiled: EXPNODE-LISPTENSOR]                | 0.33196998   | 41.441906%
<WfInst[Compiled: ADDNODE-CPUTENSOR]                 | 0.17073      | 21.313303%
<WfInst[Compiled: SCALARDIV-CPUTENSOR]               | 0.120082     | 14.990594%
<WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          | 0.068501     | 8.551412%
<WfInst[Compiled: DIVNODE-CPUTENSOR]                 | 0.050755     | 6.3360667%
<WfInst[Compiled: SUBNODE-CPUTENSOR]                 | 0.049947     | 6.2351995%
<WfInst[Compiled: <DELETED>]                         | 0.004736     | 0.5912248%
<WfInst[Compiled: SCALARMUL-CPUTENSOR]               | 0.002239     | 0.2795085%
<WfInst[Compiled: VIEWTENSORNODE-T]                  | 0.0014199999 | 0.17726754%
<WfInst[Compiled: MOVESCALARTENSORNODE-SCALARTENSOR] | 6.69e-4      | 0.08351549%
{CPUTENSOR[float] :shape (128 128) :named ChainTMP1078855 
  ((0.0017760922 0.0030971088 0.017302852  ~ 0.012318904  6.049352e-4  0.0041618845)                    
   (0.012581187  0.0030174912 0.016748475  ~ 0.007076549  0.007030908  0.0017801385)   
                 ...
   (0.0036988985 0.0061271163 0.05046869   ~ 0.009297135  0.003441493  5.820294e-4)
   (0.045387346  0.004674337  0.0018589711 ~ 0.008918608  0.0024204857 0.00761818))
  :facet :input
  :requires-grad NIL
  :backward NIL}
CL-WAFFE2-REPL>
;; (The result may not be the latest)
```
"
  (declare (type list iseq)
	   (type fixnum n-sample top-k)
	   (type boolean ignore-first-call))

  (assert (>= n-sample 1)
	  nil
	  "benchmark-accept-instructions: Assertion Failed with n-sample >= 1")

  (let* ((inst->node-table (or (third *under-benchmark-set*) (make-hash-table))) ;; NodeIID -> NodeName
	 (result)
	 (sv4bw-time 0)
	 (longest-time-width 0)
	 (total 0.0)
	 (avg 0.0)
	 (sort-by-node    (or (car *under-benchmark-set*) (make-hash-table :test #'equal)))
	 (profiled-result (or (second *under-benchmark-set*) (make-hash-table)))) ;; Basically, NodeName -> (List Time1 Time2 ...)
    (dotimes (n n-sample)
      (when iseq
	(loop with *under-benchmark-set* = (list sort-by-node profiled-result inst->node-table)
	      for inst of-type WFInstruction in iseq
	      do (let ((start-time (get-internal-real-time))) ;; Measuring save4bwtime
		   (apply-inst-sv4bw inst)
		   (let ((end-time (get-internal-real-time)))
		     (incf sv4bw-time (/ (- end-time start-time) internal-time-units-per-second))))
		     
		 (let ((start-time (get-internal-real-time)))  
		   (write-result (wfop-out-to inst) (apply-instruction inst))
		   (when (not (typep (wfop-node inst) 'function)) ;; If the node isn't codeblock...?
		     (setf (gethash (tensor-iid (wfop-self inst)) inst->node-table) inst)
		     (let* ((end-time (get-internal-real-time))
			    (executing-time (/ (- end-time start-time) internal-time-units-per-second))
			    (id (tensor-iid (wfop-self inst))))

		       (when (if ignore-first-call
				 (not (= n 0))
				 t)
			 (setf (gethash id profiled-result)
			       `(,@(gethash id profiled-result)
				 ,executing-time))))))
	      finally
		 (setq result (maybe-read-result (wfop-self inst))))))

    ;; If the execution is originated from another process
    ;; just merging the result and returns the result

    (when *under-benchmark-set*
      (return-from benchmark-accept-instructions result))

    (maphash
     #'(lambda (k v)
	 (declare (ignore k))
	 (incf avg (float (apply #'+ v)))
	 (setq longest-time-width (max longest-time-width (length (format nil "~a | " (float (apply #'+ v)))))))
     profiled-result)

    (setq total avg)
    (setq avg (/ avg (length iseq)))
    
    (flet ((conc-iseq-str (iseq)
	     (let ((tensor-ids))
	       (with-output-to-string (out)
		 (format out "[Sorted by Instructions]~%")
		 (princ " Time(s)" out)
		 (dotimes (i (abs (- longest-time-width (length " Time(s)")))) (princ " " out))
		 (format out "|   Instruction ( * - Beyonds the average execution time)~%")
		 (with-indent-to iseq
		   (loop for nth being the hash-keys in profiled-result do
		     (let ((i (gethash nth inst->node-table)))
		       (dolist (var (wfop-args i))
			 (push (tensor-id var) tensor-ids))
		       
		       ;; Put the result of benchmark
		       (let* ((times (gethash nth profiled-result))
			      (time (format nil "~a~a  "
					    (if times
						(float (apply #'+ times))
						"?")
					    (if (> (float (apply #'+ times)) avg)
						"*"
						""))))
			 
			 (when (null (gethash (instruction-opname i) sort-by-node))
			   (setf (gethash (instruction-opname i) sort-by-node) 0.0))
			 (incf (gethash (instruction-opname i) sort-by-node) (float (apply #'+ times)))
			 
			 (princ time out)
			 (dotimes (i (- longest-time-width (length time))) (princ " " out))
			 (princ "| " out)
			 (princ i out))))
		   (format out "~%~a Instructions | ~a Tensors | Overheads due to SV4BW(...) -> ~a(s) ~%"
			   (length iseq)
			   (length (remove-duplicates tensor-ids))
			   (float (/ sv4bw-time n-sample))))))))
      
      (format stream "~a~% Total Time: ~a sec~%~%" (conc-iseq-str iseq) total)

      ;; Displays top-k node
      (let* ((sort-by-node (loop for key being the hash-keys in sort-by-node
				 collect (list key (gethash key sort-by-node))))
	     (sort-by-node (sort sort-by-node #'> :key #'second))
	     (top-k        (loop for k fixnum below top-k
				 collect (nth k sort-by-node)))
	     (maxlen-node  (loop for k in top-k
				 if k
				   maximize (length (format nil "~a" (car k)))))
	     (maxtime-node  (loop for k in top-k
				  if k
				    maximize (length (format nil "~a" (second k))))))

	(setq maxtime-node (max maxtime-node (length "Total time (s)")))
	(format stream "[Sorted by topK]~%")
	(format stream "~a"
		(with-output-to-string (out)
		  (princ " Instruction" out)
		  (dotimes (i (- maxlen-node (length " Instruction"))) (princ " " out))
		  (format out " | Total time (s) | Time/Total (n-sample=~a)" n-sample)
		  (format out "~%")
		  (dolist (node top-k)
		    (when node
		      (format out  "~a" (car node))
		      (dotimes (i (- maxlen-node (length (format nil "~a" (car node))))) (princ " " out))
		      (princ " |" out)
		      (format out " ~a" (second node))
		      (dotimes (i (- maxtime-node (length (format nil "~a" (second node))))) (princ " " out))
		      (format out " | ~a%~%" (* 100 (/ (second node) total)))))))))
    result))


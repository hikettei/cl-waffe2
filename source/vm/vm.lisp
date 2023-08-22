
(in-package :cl-waffe2/vm)

(declaim (inline maybe-read-result write-result apply-instruction))
(declaim (ftype (function (AbstractTensor) AbstractTensor) maybe-read-result))
(defun maybe-read-result (tensor)
  (declare (type AbstractTensor tensor))
  (let* ((state (tensor-state tensor))
	 (res
	   (or (when state
		 (nth
		  (tensor-out-n tensor)
		  (cl-waffe2/vm.generic-tensor::statecontainer-forward-result state)))
	       tensor)))
    (the AbstractTensor res)))

(declaim (ftype (function (AbstractTensor list) t) write-result))
(defun write-result (tensor result)
  (let* ((state (tensor-state tensor)))
    ;; StateContainer should exist
    (setf (cl-waffe2/vm.generic-tensor::statecontainer-forward-result state) result)))

(declaim (ftype (function (WFInstruction) list) apply-instruction))
(defun apply-instruction (instruction)
  (declare (type WFInstruction instruction)
	   (optimize (speed 3)))
  (multiple-value-list
   (apply
    (the function (wfop-op instruction))
    (map 'list #'maybe-read-result (wfop-args instruction)))))

(declaim (ftype (function (list) (or null AbstractTensor)) accept-instructions))
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
	  do (write-result (wfop-self inst) (apply-instruction inst))
	  finally
	     (return-from accept-instructions (maybe-read-result (wfop-self inst))))))


(defun make-backward-instruction (toplevel dout-mock nth leaves fuse-p)
  (let* ((dout-input (make-input (shape dout-mock) nil
				 :create-from dout-mock
				 :scalar-p (scalar-p dout-mock)
				 :dtype (dtype dout-mock)
				 :order (order dout-mock)))
	 (bw (nth nth
		  (apply
		   #'compiler-expand-backward
		   (tensor-backward toplevel)
		   dout-input
		   (tensor-variables toplevel))))
	 (iseq (reverse (node-compile-into-vm bw :fuse-p fuse-p))))
    (when (not fuse-p)
      (apply-in-place-mutation! iseq leaves))
    (setf (tensor-state dout-input)
	  (make-statecontainer :forward-out-form (make-compiled-kernel)))
    (values
     #'(lambda (dout)
	 (declare (optimize (speed 3))
		  ;; inline accept-instructions?
		  (type AbstractTensor dout))
	 (write-result dout-input (list (maybe-read-result dout)))
	 (if iseq
	     (accept-instructions iseq)
	     dout))
     #'(lambda ()
	 (format nil "Block -> ~a-BACKWARD {
~a    }
  "
		 (class-name (class-of (tensor-backward toplevel)))
		 (with-output-to-string (out)
		   (with-indent-to iseq
		     (dolist (i iseq)
		       (let ((*node-indent* (+ 4 *node-indent*)))
			 (format out "        ~a" i))))))))))


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
CL-WAFFE2-REPL> (benchmark-accept-instructions (compile-forward-and-backward (!softmax (randn `(300 300)))) :n-sample 100)
 Time(s)   |   Instruction ( * - Beyonds the average execution time)
0.002646   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID70924 <= op(TID70924(300 300) TID70840(300 300))>
9.5e-5     | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID70937 <= op(TID70937(300 300) TID70935(300 1))>
5.4e-5     | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID70883 <= op(TID70883(300 300) TID70881(300 1))>
1.59e-4    | <WfInst[Compiled: SCALARMUL-CPUTENSOR]               : TID70843 <= op(TID70843(300 1) TID70845(1))>
4.9e-5     | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID70854 <= op(TID70854(300 300) TID70843(300 1))>
0.033278*  | <WfInst[Compiled: ADDNODE-CPUTENSOR]                 : TID70854 <= op(TID70854(300 300) TID70840(300 300))>
0.014609   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID70883 <= op(TID70883(300 300) TID70854(300 300))>
9.2e-5     | <WfInst[Compiled: MOVESCALARTENSORNODE-SCALARTENSOR] : TID70906 <= op(TID70906(1) TID70878(1))>
0.051683*  | <WfInst[Compiled: SCALARDIV-CPUTENSOR]               : TID70883 <= op(TID70883(300 300) TID70906(1))>
0.014387   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID70937 <= op(TID70937(300 300) TID70883(300 300))>
0.013283   | <WfInst[Compiled: SUBNODE-CPUTENSOR]                 : TID70924 <= op(TID70924(300 300) TID70937(300 300))>
0.002301   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID71028 <= op(TID71028(300 300) TID70924(300 300))>
0.089736*  | <WfInst[Compiled: EXPNODE-LISPTENSOR]                : TID71028 <= op(TID70924(300 300) TID71028(300 300))>
0.002164   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID71045 <= op(TID71045(300 300) TID71028(300 300))>
8.3e-5     | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID71058 <= op(TID71058(300 300) TID71056(300 1))>
1.68e-4    | <WfInst[Compiled: SCALARMUL-CPUTENSOR]               : TID70994 <= op(TID70994(300 1) TID70996(1))>
4.3e-5     | <WfInst[Compiled: VIEWTENSORNODE-T]                  : TID71005 <= op(TID71005(300 300) TID70994(300 1))>
0.002178   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID70977 <= op(TID70977(300 300) TID70924(300 300))>
0.089359*  | <WfInst[Compiled: EXPNODE-LISPTENSOR]                : TID70977 <= op(TID70924(300 300) TID70977(300 300))>
0.033492*  | <WfInst[Compiled: ADDNODE-CPUTENSOR]                 : TID71005 <= op(TID71005(300 300) TID70977(300 300))>
0.014615   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID71058 <= op(TID71058(300 300) TID71005(300 300))>
0.014399   | <WfInst[Compiled: DIVNODE-CPUTENSOR]                 : TID71045 <= op(TID71045(300 300) TID71058(300 300))>

22 Instructions | 19 Tensors
 Instruction                                         | Total time (s) (n-sample=100)
<WfInst[Compiled: EXPNODE-LISPTENSOR]                | 0.179095
<WfInst[Compiled: ADDNODE-CPUTENSOR]                 | 0.06677
<WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          | 0.052899994
<WfInst[Compiled: SCALARDIV-CPUTENSOR]               | 0.051683
<WfInst[Compiled: DIVNODE-CPUTENSOR]                 | 0.014399
<WfInst[Compiled: SUBNODE-CPUTENSOR]                 | 0.013283
<WfInst[Compiled: SCALARMUL-CPUTENSOR]               | 3.27e-4
<WfInst[Compiled: VIEWTENSORNODE-T]                  | 3.24e-4
<WfInst[Compiled: MOVESCALARTENSORNODE-SCALARTENSOR] | 9.2e-5
```
"
  (declare (type list iseq)
	   (type fixnum n-sample top-k)
	   (type boolean ignore-first-call))

  (assert (>= n-sample 1)
	  nil
	  "benchmark-accept-instructions: Assertion Failed with n-sample >= 1")

  (let ((result)
	(longest-time-width 0)
	(total 0.0)
	(avg 0.0)
	(sort-by-node    (make-hash-table :test #'equal))
	(profiled-result (make-hash-table))) ;; Basically, NodeName -> (List Time1 Time2 ...)
    (dotimes (n n-sample)
      (when iseq
	(loop for inst of-type WFInstruction in iseq
	      for nth fixnum upfrom 0
	      do (let ((start-time (get-internal-real-time)))
		   (write-result (wfop-self inst) (apply-instruction inst))
		   (let* ((end-time (get-internal-real-time))
			  (executing-time (/ (- end-time start-time) internal-time-units-per-second)))

		     (when (if ignore-first-call
			       (not (= n 0))
			       t)
		       (setf (gethash nth profiled-result)
			     `(,@(gethash nth profiled-result)
			       ,executing-time)))))
	      finally
		 (setq result (maybe-read-result (wfop-self inst))))))

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
		 (princ " Time(s)" out)
		 (dotimes (i (abs (- longest-time-width (length " Time(s)")))) (princ " " out))
		 (format out "|   Instruction ( * - Beyonds the average execution time)~%")
		 (with-indent-to iseq
		   (loop for i in iseq
			 for nth fixnum upfrom 0 do
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
		 (format out "~%~a Instructions | ~a Tensors~%"
			 (length iseq)
			 (length (remove-duplicates tensor-ids)))))))
      
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


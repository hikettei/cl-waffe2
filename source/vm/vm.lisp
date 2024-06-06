
(in-package :cl-waffe2/vm)

(defparameter *opt-level* 2 "
## [parameter] `*opt-level*`

This parameter indicates the degree of runtime error detection. Whichever you choose, cl-waffe2 never apply something unsafe code transformation. It takes the fixnum from 1 to 3, and the larger, the faster.

- Set 1 to use safety-mode, in every instructions, runtime error is checked.

- Set 2 to use middle-mode, runtime error is checked only when first execution.

- Set 3 to use fastest-mode, no runtime error checking is done.

Again, whichever levels you choose, the graph cl-waffe2 executes is the same. So the effects on the performance is very small (within < `1e-4~1e-5` sec).

In default, set to 2.
")

(declaim (type (integer 0 3) *opt-level*))

(defparameter *logging-vm-execution* nil "
## [parameter] `*logging-vm-execution*`

This parameter is useful for printing how all instructions are performed. If set to T, all results and arguments produced by executing `cl-waffe2 IR` is displayed into the terminal. In default, set to nil.
")

(declaim (inline maybe-read-result write-result apply-instructions apply-inst-sv4bw))
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
	      ;; Intentionally creates the illusion of VM that misunderstands
	      ;; var is [computed] by deleting tensor-state
	      (tensor-vec place)
	      (tensor-vec var)

	      (if (or (scalar-p place) (scalar-p var))
		  (setf (tensor-vec place) (tensor-vec var))
		  (let* ((s1  (shape place))
			 (s2  (shape var))
			 (s1a (actual-shape place))
			 (s2a (actual-shape var)))
		    (setf (cl-waffe2/vm.generic-tensor::tensor-visible-shape place) s1a
			  (cl-waffe2/vm.generic-tensor::tensor-visible-shape var) s2a)
		    (unwind-protect (%vm-move place var)
		      (setf (cl-waffe2/vm.generic-tensor::tensor-visible-shape place) s1
			    (cl-waffe2/vm.generic-tensor::tensor-visible-shape var) s2)))))))
  nil)

(declaim (ftype (function (AbstractTensor) AbstractTensor) maybe-read-result))
(defun maybe-read-result (tensor)
  (declare (type AbstractTensor tensor))
  (if (tensor-tmp-p tensor)
      (let ((out (read-from-mempool-tensor tensor)))
	;; Keep Broadcasting, Permution etc... But storages are shared.
	(setf (tensor-vec tensor) (cl-waffe2/vm.generic-tensor::vec out))
	tensor)
      (if (scalar-p tensor)
	  tensor
	  (let* ((state (tensor-state tensor))
		 (res
		   (or (when state
			 (cl-waffe2/vm.generic-tensor::statecontainer-forward-result state))
		       tensor)))
	    ;;(setf (tensor-initial-offset res) (tensor-initial-offset tensor))
	    (the AbstractTensor res)))))

(declaim (ftype (function (list list) t) write-result))
(defun write-result (tensors results)
  ;; [TODO] Runtime Shape-Error Detection
  (loop for tensor of-type AbstractTensor in tensors
	for result in results
	if  (and tensor result) do
	  (if (tensor-tmp-p tensor)
	      (progn
		;; tensor ... out-to
		;; results ... result
		;; Tensors registerd in the memory-pool,
		;; doesn't need the support of StateContainer anymore
		;; Deleting Unused StateContainer will benefit:
		;;  The returned tensor by the proceed function is
		;;  displayed as [computed] in the terminal.
		;; ScalarTensors never use Memory-Pool
		;; Update Memory-Pool
		;;(setf (tensor-vec (read-from-mempool-tensor tensor)) (cl-waffe2/vm.generic-tensor::vec result))
		(update-alloc-vec! tensor result)
		;; Tensor is already broadcasted/permuted...
		;; So sharing vec is enough.
		(setf (tensor-vec tensor) (cl-waffe2/vm.generic-tensor::vec result))
		;;(cl-waffe2/vm.generic-tensor::embody-tensor-vec tensor result)
		) ;; Tensor.ID <- Result
	      (if (scalar-p tensor)
		  (setf (tensor-vec tensor) (tensor-vec result))
		  (let* ((state (tensor-state tensor)))
		    (setf (cl-waffe2/vm.generic-tensor::statecontainer-forward-result state) result))))))

(declaim (ftype (function (WFInstruction) list) apply-instruction))
(defun apply-instruction (instruction)
  (declare (type WFInstruction instruction)
	   (optimize (speed 3)))

  ;; [TODO] Bring it to the toplevel?
  (when (<= *opt-level* 2)
    (node-realize-assertions (wfop-node instruction)))
  
  (when *logging-vm-execution*
    (let* ((inst (format nil "~a" instruction))
	   (cnt  (length inst)))
      (format t "= [*logging-vm-execution*] ~a
Instruction: ~a
[Inputs]
~a"
	      (with-output-to-string (out)
		(dotimes (i cnt) (princ "=" out)))
	      inst
	      (if (some #'(lambda (x) (not (every #'numberp (tensor-stride x)))) (map 'list #'maybe-read-result (wfop-args instruction)))
		  (with-output-to-string (out)
		    (format out "(Can't Displayed):")
		    (dolist (arg (map 'list #'maybe-read-result (wfop-args instruction)))
		      (format out " ~a" (cl-waffe2/vm.nodes::describe-tensor arg))))
		  (map 'list #'maybe-read-result (wfop-args instruction))))))

  (let ((outs (multiple-value-list
	       (apply
		(the function (wfop-op instruction))
		(map 'list #'maybe-read-result (wfop-args instruction))))))

    (when (or (null outs)
	      (not (every #'(lambda (x) (typep x 'AbstractTensor)) outs)))
      (error "cl-waffe2 VM: Runtime Error.
The instruction ~a returned an invalid typed result.
All outputs must be an AbstractTensor, but it returned: ~a"
	     instruction
	     outs))
    
    (when (and (< *opt-level* 3)
	       (wfop-self instruction)
	       (tensor-backward (wfop-self instruction))
	       (if (= *opt-level* 1)
		   t
		   (null (wfop-error-check-p instruction)))
	       (not (ignore-shape-error (tensor-backward (wfop-self instruction)))))

      (when (and (= *opt-level* 1)
		 (not (= (length outs) (length (wfop-out-to instruction)))))
	(warn "cl-waffe2 VM: Runtime Warning
The instruction: ~a
should return ~R arguments, but provided ~R.

out-to returned:

~a"
	      instruction
	      (length outs)
	      (length (wfop-out-to instruction))
	      outs))
      
      (mapc #'(lambda (expected received)
		(when (not (eql (the boolean (scalar-p expected)) (scalar-p received)))
		  (warn "cl-waffe2 VM: Runtime Warning
The instruction: ~a
~a
Expected:
~a
But got:
~a"
			instruction
			(if (scalar-p received)
			    "returned a ScalarTensor but Matrix is expected:"
			    "returned a Matrix but ScalarTensor is expected:")
			expected
			received))

		;; Under *opt-level* = 1, we do runtime shape inspection
		;; to all the tensors

		;; Under *opt-level* = 2, weodo runtime shape inspection to:
		;;  1. Tensors with fixed shape
		;;  2. This is done only the first time.
		(when (if (= *opt-level* 1)
			  (not ;; Shape-Equal is slow op
			   (cl-waffe2/vm.generic-tensor::shape-equal
			    (shape expected) (shape received)))
			  (not
			   (or (some #'symbolp (shape expected))
			       (some #'symbolp (shape received))
			       (equal (shape expected) (shape received)))))
		  (warn "cl-waffe2 VM: Runtime Warning
The instruction: ~a
Shapes are incompatible.
Expected:
~a
But got:
~a"
			instruction
			expected
			received)))
	    (wfop-out-to instruction) outs))
    
    (setf (wfop-error-check-p instruction) T)

    (when *logging-vm-execution*
      (format t "
outs:
~a"
	      (if (some #'(lambda (x) (not (every #'numberp (tensor-stride x)))) outs)
		  (with-output-to-string (out)
		    (format out "(Can't Displayed):")
		    (dolist (arg outs)
		      (format out " ~a" (cl-waffe2/vm.nodes::describe-tensor arg))))
		  outs)))
    outs))

(defun runtime-error (position condition iseq)
  (error "cl-waffe2 VM: Encountered Runtime Error at ~ath instruction.
disassemble:
~a

condition:
  ~a

~a"
	 position
	 (with-output-to-string (out)
	   (let* ((start (max 0 (- position 3)))
		  (end   (min (length iseq) (+ position 3)))
		  (iseqs  (loop for nth upfrom start below end collect (nth nth iseq))))
	     (with-indent-to iseqs
	       (loop with *no-newline* = t
		     for nth upfrom start below end
		     if (= nth position)
		       do (format out "~a*: ~a~%" nth (nth nth iseq))
		     else
		       do (format out "~a : ~a~%" nth (nth nth iseq))))))
	 condition
	 (render-debug-info)))

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
	  for position fixnum upfrom 0
	  do (apply-inst-sv4bw inst)
	     (handler-bind
		 ((error
		    (lambda (c)
		      (runtime-error position c iseq))))
	       (write-result (wfop-out-to inst) (apply-instruction inst)))
	  finally
	     (return-from accept-instructions
	       (apply #'values
		      (map 'list
			   (compose
			    #'cl-waffe2/vm.nodes::eliminate-undetermined-size
			    #'maybe-read-result)
			   (wfop-out-to inst)))))))

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

### Inputs

`n-sample[fixnum]` repeats the iseq execution for `n-sample` times

`ignore-first-call[boolean]` If t, ignores the first call to avoid including allocating time.

`stream[stream]` the place to display the result

`top-k[fixnum]` top-k slowest nodes are displayed at the end of report.

### Return

`result[AbstractTensor]`

See also: `proceed-bench`
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
	      for position fixnum upfrom 0
	      do (let ((start-time (get-internal-real-time))) ;; Measuring save4bwtime
		   (apply-inst-sv4bw inst)
		   (let ((end-time (get-internal-real-time)))
		     (incf sv4bw-time (/ (- end-time start-time) internal-time-units-per-second))))
		     
		 (let ((start-time (get-internal-real-time)))
		   (handler-bind
		       ((error
			  (lambda (c)
			    (runtime-error position c iseq))))
		     (write-result (wfop-out-to inst) (apply-instruction inst)))
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


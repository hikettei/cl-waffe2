
(in-package :cl-waffe2/vm.iterator)

;; TODO: Producing a list of subscript, iterations from DSL like:
;; In the future release, convnode, im2col is removed since the parallelization
;; is depends on it

;;
;; for (i=0..1)
;;     for (j=0..2)
;;         out[ik] += x[ij] * y[jk] (OP = einsum:mul_add)
;;

;; for (i=0.10)
;;  for (j=0.10)
;;   for (k=0.10)
;;    out[ik] += x[ij] * y[jk]

;; TO ADD: simdpack, unpack

;; TODO: Producing a Lisp-Like DSL
;;        -> GCC, Metal, CUDA, Lisp etc...
;;        -> Einsum notation
;;

;; Continuous iterators are foled
;; reduction
;; Scheduler -> Lisp-Like AST -> CUDA. GCC, Lisp, Metal etc...

(defvar *dependency-graph*)
(defun make-dependency-graph (actions)
  (declare (type list actions)
	   (optimize (speed 3)))
  (let ((out (make-hash-table :test #'eql)))
    (dolist (act (reverse actions))
      (dolist (src (action-source act))
	(dolist (tgt (action-target act))
	  (let ((key (tensor-iid (ispace-tensor src)))
		(val (tensor-iid (ispace-tensor tgt))))
	    (if (gethash key out)
		(push val (gethash key out))
		(setf (gethash key out) `(,val)))))))
    out))

(defun a-depends-b? (a b)
  "Tracing the dependency graph from a, the function returns T if there's b otherwise nil."
  (declare (type (or keyword symbol) a b)
	   (optimize (speed 3)))
  (labels ((helper (value)
	     (let ((deps (gethash value *dependency-graph*)))
	       (dolist (dep deps)
		 (when (eql dep b) (return-from a-depends-b? t))
		 (helper dep)))))
    (helper a)))

(defun dependent-p (act1 act2)
  "Returns T if one of src/tgt in act2 depends on the tgt of act1"
  (let ((act1-t (action-target act1))
	(act2-s (action-source act2))
	(act2-t (action-target act2)))
    (flet ((find-helper (tensors &aux (tensors (map 'list #'ispace-tensor tensors)))
	     (dolist (tgt act1-t)
	       (let ((tgt (tensor-iid (ispace-tensor tgt))))
		 (dolist (tensor tensors)
		   (when (a-depends-b?
			  ;; Tracing the grpah from tensor
			  ;; and tgt found, it returns t.
			  (tensor-iid tensor)
			  tgt)
		     (return-from dependent-p t)))))))
      
      (find-helper act2-s)
      (find-helper act2-t))))

(defun generate-indices (n-rank)
  (declare (type fixnum n-rank))
  (loop for n upfrom 0 below n-rank
	collect
	(intern (format nil "GID~a" n))))

(defun trace-invocation (op source-tensor target-tensor &key (kernel-rank 1) (collapse t))
  "call-with-view -> invocation
Assertion: Shapes are already determined."
  (declare (ignore collapse))
  (assert (= kernel-rank 1) () "kernel-rank must be 1")
  (assert (= (dims source-tensor) (dims target-tensor)) () "Assertion failed with ~a and ~a should have the same ranks" source-tensor target-tensor)

  (let ((source
	  (make-indexspace
	   source-tensor
	   :subscripts (generate-indices (dims source-tensor))
	   :sizes (wf/t::translate-adjustable-shape (shape source-tensor))))
	(target
	  (make-indexspace
	   target-tensor
	   :subscripts (generate-indices (dims target-tensor))
	   :sizes (wf/t::translate-adjustable-shape (shape target-tensor)))))
    (make-action
     :source (list source)
     :target (list target)
     :op op
     :rank (dims source-tensor)
     :depends (find-depends source target))))

(defun make-invocation (op sources targets)
  "Creates an action manually; sources and targets are given as a list."
  (make-action
   :source sources
   :target targets
   :op op
   :rank (dims (ispace-tensor (car sources)))
   :depends (apply #'find-depends `(,@sources ,@targets))))

(defstruct IterStage ;; Corresponds with dotimes
  (determines nil :type (or symbol keyword)) ;; A list of symbols which this stage determines
  (size       0   :type fixnum)
  (rank       0   :type fixnum)
  (ops        nil :type list)

  (parallel    nil :type (or null fixnum))
  (tiling      nil :type list)
  (simd-unroll nil :type (or null fixnum))   ;; simd-unroll * stride + reminder = size
  (simd-reminder nil :type (or null fixnum)) ;; 
  (unroll      nil :type (or null fixnum))
  (reduction   nil :type boolean))

(defstruct Scheduler
  "args ... ((:in or :out or :io tensor) (:in or :out or :io tensor) ...)"
  (name  :plain  :type keyword)
  (args  nil :type list)
  (iters nil :type list)) ;; A list of IterStage

(defun schedule-name! (scheduler)
  "Returns a string including a list of ops (with shaped). it can be used to reuse compiled function."
  (declare (type scheduler scheduler))
  (setf (scheduler-name scheduler)
	(intern
	 (with-output-to-string (out)
	   (format out "~(~a~)_" (car wf/t:*using-backend*))
	   (let ((stages (sort-stage scheduler)))
	     (dolist (stgs stages)
	       (dolist (stg stgs)
		 (dolist (op (iterstage-ops stg))
		   (format out "~(~a~)_"
			   (car (cl-ppcre:split "-" (format nil "~a" (action-op op)))))
		   (dolist (src (action-source op))
		     (format out "~(~a~)_" (dtype (ispace-tensor src)))
		     (dolist (s (shape (ispace-tensor src)))
		       (format out "~a_" s)))
		   (dolist (tgt (action-target op))
		     (format out "~(~a~)_" (dtype (ispace-tensor tgt)))
		     (dolist (s (shape (ispace-tensor tgt)))
		       (format out "~a_" s)))))))
	   (format out "fused"))
	 "KEYWORD")))

(defun schedule-solve-tensor-dependencies! (scheduler &aux (sorted (sort-stage scheduler)))
  (declare (type Scheduler scheduler))
  (let* ((ops  (loop for stages in sorted
		     append
		     (loop for stage in stages
			   append
			   (iterstage-ops stage))))
	 (all-tensors
	   (loop for op in ops
		 append
		 `(
		   ,@(loop for src in (action-source op)
			   append
			   (list (ispace-tensor src)))
		   ,@(loop for tgt in (action-target op)
			   append
			   (list (ispace-tensor tgt))))))
	 (all-tensors
	   (remove-duplicates all-tensors :test #'eql :key #'tensor-id))
	 (deps (make-hash-table :test #'eql)))

    (loop for op in ops do
      (flet ((apply-helper (list default not)
	       (loop for src in list
		     for id = (tensor-id (ispace-tensor src)) do
		       (when (and
			      (gethash id deps)
			      (eql (gethash id deps) default))
			 (setf (gethash id deps) :io))
		       (alexandria:ensure-gethash
			id
			deps
			not))))
	(apply-helper (action-source op) :out :in)
	(apply-helper (action-target op) :in :out)))
    
    (setf (scheduler-args scheduler)
	  (map
	   'list
	   #'(lambda (tensor)
	       (cons (gethash (tensor-id tensor) deps) tensor))
	   all-tensors))))

(defun sort-stage (scheduler)
  (let* ((stages (scheduler-iters scheduler))
	 (rank   (1+ (apply #'max (map 'list #'iterstage-rank stages))))
	 (out    (make-list rank)))
    (dolist (stage stages)
      (let ((rank (iterstage-rank stage)))
	(setf (nth rank out) `(,@(nth rank out) ,stage))))
    out))

(defparameter *indentation* 0)
(defmethod print-object ((iter IterStage) stream)
  (flet ((indent (out)
	   (dotimes (i *indentation*) (princ " " out))))
    (macrolet ((of (x) `(,(symb 'iterstage- x) iter)))
      (format
       stream
       "~a"
       (with-output-to-string (out)
	 (when (of parallel)
	   (indent out)
	   (format out "@parallel(~a)~%" (of parallel)))
	 (when (of tiling)
	   (indent out)
	   (format out "@tiling~a~%" (of tiling)))
	 (when (of unroll)
	   (indent out)
	   (format out "@unroll~a~%" (of unroll)))
	 (when (of simd-unroll)
	   (indent out)
	   (format out "@SIMDified(~a*~a + ~a)~%"
		   (of simd-unroll)
		   (floor (/ (of size) (of simd-unroll)))
		   (of simd-reminder)))
	 (when (of reduction)
	   (indent out)
	   (format out "@reduction~%"))
	 (indent out)
	 (format out "dotimes ~a=0..~a~%"
		 (of determines)
		 (of size))
	 (let ((*indentation* (+ *indentation* 2)))
	   (flet ((print-action (action
				 &aux (rank (action-rank action)))
		    (indent out)
		    (format out "~a: " (action-op action))
		    (dolist (tgt (action-target action))
		      (format out "~a[" (tensor-id (ispace-tensor tgt)))
		      (loop for ref in (ispace-space tgt)
			    for c upfrom 0 do
			      (when (and
				     (not (= c 0))
				     (not (= rank 0)))
				(format out "+"))
			      (print-iref ref out))
		      (format out "]"))
		    (format out " <- ")
		    (dolist (tgt (action-source action))
		      (format out "~a[" (tensor-id (ispace-tensor tgt)))
		      (loop for ref in (ispace-space tgt)
			    for c upfrom 0 do
			      (when (and
				     (not (= c 0))
				     (not (= rank 0)))
				(format out "+"))
			      (print-iref ref out))
		      (format out "] "))
		    (fresh-line out)))
	     (mapc #'print-action (iterstage-ops iter))
	     (when (null (iterstage-ops iter)) (fresh-line out)))))))))

(defmethod print-object ((scheduler Scheduler) stream)
  (let ((sorted (sort-stage scheduler)))
    (format
     stream
     "Scheduler: ~a~%  (~a)~%~a"
     (scheduler-name scheduler)
     (with-output-to-string (out)
       (loop for (io . arg) in (scheduler-args scheduler)
	     for nth upfrom 0
	     for lastp = (= nth (1- (length (scheduler-args scheduler)))) do
	       (format out "(~a ~(~a~) ~a)~a"
		       io
		       (dtype arg)
		       (tensor-id arg)
		       (if lastp "" " "))))
     (with-output-to-string (out)
       (loop for stages in sorted
	     for rank upfrom 0
	     for *indentation* = (1+ (* rank 2)) do
	       (dolist (stage stages)
		 (format out "~a" stage)))))))

(defun schedule-fuse (schedule1 schedule2)
  "Fuses the given two schedules.
If two schedules can be fused, returns a new schedule otherwise nil.
Schedules that can be fused is defined as:
- Iterators, ranks, and sizes are the same."
  (declare (type Scheduler schedule1 schedule2)
	   (optimize (speed 3)))
  (symbol-macrolet ((->failed (return-from schedule-fuse (values schedule1 schedule2))))
    (let ((sorted1 (sort-stage schedule1))
	  (sorted2 (sort-stage schedule2)))
      (declare (type list sorted1 sorted2))
      ;; Mismatched ranks
      (when (not (= (length sorted1) (length sorted2)))->failed)
      (make-scheduler
       :iters
       (loop for stages1 list in sorted1
	     for stages2 list in sorted2
	     for rank fixnum upfrom 0
	     collect
	     (flet ((fuse-stg (stg1 stg2)
		      ;; Different iterations cannot be fused
		      (when (not (= (iterstage-size stg1) (iterstage-size stg2)))->failed)		   
		      ;; All iterators indicates the same one.
		      (when (not (eql (iterstage-determines stg1) (iterstage-determines stg2)))->failed)
		      ;; Here, if stg1 and stg2 are independent
		      ;; the order can be swapped
		      ;; otherwise split the loops
		      ;; note that stg1 comes first and stg2 follows in the wengert list
		      ;; If one of src/tgt in stg2 depends on stg2, they cannot be swapped

		      ;; A   B
		      ;;  \ /
		      ;;   C  D  
		      ;;   \ /
		      ;;    E
		      ;; E depends on A, B, C
		      ;; But with topologically sorted, D could be combined with one of A, B, C, D
		      ;; C, and E has the equivalent iterator and fused either of one.

		      ;; TODO: As for independent nodes
		      ;; I think we can do more;
		      ;; Can MoveTensorNode be fused together (based on dependency graph)?
		      (make-iterstage
		       :determines (iterstage-determines stg1)
		       :size       (iterstage-size stg1)
		       :rank       (iterstage-rank stg1)
		       :reduction  (or (iterstage-reduction stg1) (iterstage-reduction stg2))
		       ;; [TODO]: Fuse MoveTensorNode !!
		       :ops `(,@(iterstage-ops stg1)
			      ,@(iterstage-ops stg2)))))
	       ;; Fusion is performed in the first process of compiling
	       ;; so it doesnt handle with complicated iterators
	       (when (or (not (= 1 (length stages1))) (not (= 1 (length stages2))))->failed)
	       (fuse-stg (car stages1) (car stages2))))))))

(defun schedule-reorder (schedule orders)
  "Shuffles the order of schedule given new-orders
Returns:
Scheduler whose iterators are shuffled following:
    new-orders = (schedule[orders_0] schedule[orders_0] schedule[orders_2])"
  (declare (type Scheduler schedule)
	   (type list orders))
  (let* ((sorted (sort-stage schedule))
	 (ops  (loop for stages in sorted
		     append
		     (loop for stage in stages
			   append
			   (iterstage-ops stage))))
	 (ops-place-to-rank (car (last orders))))	   
    
    (assert (= (length orders)
	       (length sorted))
	    ()
	    "Assertion failed, before and after the reordering, the rank should correspond with")    
    
    (make-scheduler
     :iters
     (loop for ref in orders
	   for nth upfrom 0
	   for stages = (find ref sorted :key (alexandria:compose #'iterstage-determines #'car) :test #'eql)
	   append
	   (loop for stage in stages
		 collect
		 (let ((stage (copy-iterstage stage)))
		   (setf (iterstage-rank stage) nth
			 (iterstage-ops  stage) nil
			 (iterstage-determines stage) ref)
		   (when (eql ref ops-place-to-rank)
		     (setf (iterstage-ops stage) ops))
		   stage))))))


(defun schedule-parallelize! (schedule rank n-threads)
  (declare (type fixnum rank)
	   (type Scheduler schedule))
  (when (null n-threads) (return-from schedule-parallelize!))

  ;; Reading the dependency return nil if it is impossible to parallelize
  (let ((sorted (sort-stage schedule)))
    (mapc
     #'(lambda (iter)
	 (when (> (iterstage-size iter) n-threads)
	   (setf (iterstage-parallel iter) n-threads)))
     (nth rank sorted))))

(defun schedule-bind! (schedule rank name)
  (declare (type Scheduler schedule)
	   (type fixnum rank)
	   (type string name))
  (error "not implemented")
  )

;;(defun schedule-tiling! (schedule rank tiles))
;;(defun schedule-unroll! (schedule rank n))

(defun schedule-simdify! (schedule stride)
  "Inserts the simdified operations in the last axis"
  ;; pack unpack
  (declare (type Scheduler schedule)
	   (type fixnum stride))
  (let* ((sorted (sort-stage schedule)))
    (let ((last-stages (car (last sorted))))
      (mapc
       #'(lambda (stage)
	   (when (>= (iterstage-size stage) stride)
	     (setf (iterstage-simd-unroll stage) stride
		   (iterstage-simd-reminder stage) (mod (iterstage-size stage) stride))))
       last-stages))))

(defun create-schedule (actions)
  (flet ((%make-scheduler (action
			   &aux
			     (indices
			      `(,@(action-source action)
				,@(action-target action)))
			     (refs
			      (apply
			       #'find-depends-symbol
			       indices)))
	   (make-scheduler
	    :iters
	    (loop for ref in refs
		  for info-tensor = (find ref indices
					  :test
					  #'(lambda (index space)
					      (find index space :test #'eql :key #'iref-index))
					  :key #'ispace-space)
		  for info = (find ref (ispace-space info-tensor) :key #'iref-index)
		  for n upfrom 0
		  collect
		  (make-iterstage
		   :determines (iref-index info)
		   :size       (iref-size info)
		   :rank       n
		   :parallel   nil
		   :unroll     nil
		   :tiling     nil
		   :reduction  nil
		   :ops (when (eql ref (car (last refs)))
			  (list action)))))))
    ;; First, The Compiler is eager tog fuse as many ops as possible
    (let ((schedules (map 'list #'%make-scheduler actions)))
      (flet ((fuse-helper (&rest args)
	       (multiple-value-bind (old new) (apply #'values args)
		 (if (null new)
		     old
		     (if (listp old)
			 `(,@(butlast old)
			   ,@(multiple-value-list (schedule-fuse (car (last old)) new)))
			 (multiple-value-list (schedule-fuse old new)))))))
	(when (> (length schedules) 1)
	  (setf schedules (reduce #'fuse-helper schedules))))
      schedules)))

(defun solve-actions (actions
		      &key
			(n-threads nil)
			(simd-stride nil) ;; <- should be nil in default
			)
  "Receives invocations (A set of actions)"
  (let* ((schedules (create-schedule actions)))
    ;; Minimizes the loss
    (flet ((optimize-helper (schedule)
	     ;; Simdifiy . Collapse . Optimize
	     (let ((parallelized (polyhedral-optimize schedule n-threads)))
	       ;; Loop Collapse
	       (schedule-collapse! parallelized)
	       (when simd-stride
		 (schedule-simdify! parallelized simd-stride))
	       (schedule-name! parallelized)
	       (schedule-solve-tensor-dependencies! parallelized)
	       parallelized)))
      (map 'list #'optimize-helper schedules))))

(defgeneric schedule-codegen (backend-indicator Scheduler)
  (:documentation
   "
## [generic] schedule-codegen

```lisp
(schedule-codegen backend-indicator Scheduler)
```

"))

(defgeneric schedule-config (backend-indicator)
  (:documentation
   "
## [generic] schedule-config

```lisp
(schedule-config backend-indicator)
;; -> (values SIMD-Stride, N-Threads)
```
"))

(defmethod schedule-config (backend-indicator) (values nil nil))

;; TODO: Full example of using cl-waffe2 scheduler

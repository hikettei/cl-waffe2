
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

;; [Memo]
;; 同じサイズのIter同士をFuse
;; CollapseできるものはCollapseで対処
;; 線形計画法でSIMD Pack/Unpack, OpenMP, Memory Locality

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
	(intern (format nil "GID~a" n) "KEYWORD")))

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

(defstruct IterStage ;; Corresponds with dotimes
  (determines nil :type (or symbol keyword)) ;; A list of symbols which this stage determines
  (size       0   :type fixnum)
  (rank       0   :type fixnum)
  (ops        nil :type list)

  (parallel    nil :type (or null fixnum))
  (tiling      nil :type list)
  (unroll      nil :type (or null fixnum))
  (reduction   nil :type boolean))

(defstruct Scheduler
  (iters nil :type list)) ;; A list of IterStage

(defun scheduler-name (scheduler)
  "Returns a string including a list of ops (with shaped). it can be used to reuse compiled function."
  (declare (type scheduler scheduler))

  )

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
	 (when (of reduction)
	   (indent out)
	   (format out "@reduction~%"))
	 (indent out)
	 (format out "dotimes ~a=0..~a~%"
		 (of determines)
		 (of size))
	 (let ((*indentation* (+ *indentation* 2)))
	   (indent out)
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
		      (format out "]~%"))))
	     (fresh-line out)
	     (mapc #'print-action (iterstage-ops iter))
	     (when (null (iterstage-ops iter)) (fresh-line out)))))))))

(defmethod print-object ((scheduler Scheduler) stream)
  (let ((sorted (sort-stage scheduler)))
    (format
     stream
     "Scheduler:~%~a"
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
	 (dims (loop for rank upfrom 0 below (length sorted) collect rank))
	 (ops  (loop for stages in sorted
		     append
		     (loop for stage in stages
			   append
			   (iterstage-ops stage))))
	 (sorted-order
	   (loop for order in orders
		 collect
		 (nth order dims)))
	 (symbols (loop for rank in sorted
			collect
			(iterstage-determines (car rank))))
	 (last-rank (iterstage-rank (caar (last sorted)))))
    
    (assert (= (length orders)
	       (length sorted))
	    ()
	    "Assertion failed, before and after the reordering, the rank should correspond with")
    
    (make-scheduler
     :iters
     (loop for ref   in orders
	   for order in sorted-order
	   for nth upfrom 0
	   for stages = (nth nth sorted)
	   append
	   (loop for stage in stages
		 collect
		 (let ((stage (copy-iterstage stage)))
		   (setf (iterstage-rank stage) ref
			 (iterstage-ops  stage) nil
			 (iterstage-determines stage) (nth nth symbols))
		   (when (= order last-rank)
		     (setf (iterstage-ops stage) ops))
		   stage))))))


(defun schedule-parallelize (schedule rank n-threads)
  (declare (type fixnum rank n-threads)
	   (type Scheduler schedule))

  ;; Reading the dependency return nil if it is impossible to parallelize
  
  (let ((sorted (sort-stage schedule)))
    (mapc
     #'(lambda (iter)
	 (setf (iterstage-parallel iter) n-threads))
     (nth rank sorted))))

(defun schedule-bind! (schedule rank name)
  (declare (type Scheduler schedule)
	   (type fixnum rank)
	   (type string name))
  
  )

;;(defun schedule-tiling! (schedule rank tiles))
;;(defun schedule-unroll! (schedule rank n))

(defun schedule-simdify (schedule rank stride)
  "Splits the iteration at the rank by `stride`"
  
  )

(defun create-schedule (actions)
  (flet ((%make-scheduler (action)
	   (make-scheduler
	    :iters
	    (loop for ref in (ispace-space (car (action-target action)))
		  for n upfrom 0
		  collect
		  (make-iterstage
		   :determines (iref-index ref)
		   :size       (iref-size ref)
		   :rank n
		   :parallel nil
		   :unroll nil
		   :tiling nil
		   :reduction (= (iref-stride ref) 0)
		   :ops (when (= n (1- (action-rank action)))
			  (list action)))))))
    ;; First, The Compiler is eager to fuse as many ops as possible
    (let ((schedules (map 'list #'%make-scheduler actions)))
      (flet ((fuse-helper (&rest args)
	       (multiple-value-bind (old new) (apply #'values args)
		 (if (null new)
		     old
		     (if (listp old)
			 `(,@(butlast old)
			   ,@(multiple-value-list (schedule-fuse (car (last old)) new)))
			 (multiple-value-list (schedule-fuse old new)))))))
	(setf schedules (reduce #'fuse-helper schedules)))
      schedules)))

(defun solve-invocations (invocations)
  "Receives invocations (A set of actions)"
  (let* ((*dependency-graph* (make-dependency-graph invocations))
	 (schedules (create-schedule invocations)))

    ;; Minimizes the loss
    (map 'list #'polyhedral-optimize! schedules)))

;; TODO: Full example of using cl-waffe2 scheduler

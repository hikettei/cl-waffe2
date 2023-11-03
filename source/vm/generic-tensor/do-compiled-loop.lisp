
(in-package :cl-waffe2/vm.generic-tensor)

;;
;; do-compiled-loop.lisp explores all possible routes when expanding ranked matrix operations
;; and minimize the costs and maximize the use of SIMD register
;; This computation will be done in runtime << 1e-5 sec and cached.
;; In the future release, integrate this file into call-with-view for simplicity
;; Reading List:
;;  Ref: https://inria.hal.science/inria-00551077/document
;;       https://atrg.jp/ja/index.php?plugin=attach&pcmd=open&file=20171007_ATOS17_Sato.pdf&refer=ATOS17
;;       https://arxiv.org/pdf/2005.04091.pdf
;;       http://perso.ens-lyon.fr/christian.perez/_media/180308/cash_cohen.pdf
;;       Polyhedral Compiler
;;       https://ucbrise.github.io/cs294-ai-sys-sp19/assets/lectures/lec12/dl-compilers.pdf
;;       https://arxiv.org/pdf/2005.04091.pdf
;;

;;
;; As of this writing, features on iterations works enough as for element-wise operations
;; but as for permuted tensors, it signifcantly reduces the performance.
;; We can easily tackle this problem by using foreign DL Frameworks like oneDNN; but it restricts the flexibility of cl-waffe2
;; Loop Oriented Optimization should not be limited to call foreign libraries; implement kernel-size=0 and
;; JIT Compiling to Vectorized C++/CUDA Kernel?
;;

(defstruct (AbstractLoop
	    (:conc-name aloop-)
	    (:constructor make-aloop (rank by element-n size mode)))
  (rank rank :type fixnum)
  (by   by   :type (or symbol fixnum))
  (size size :type (or fixnum list symbol)) ;; (loop for i ...
  (element-n element-n :type (or null symbol fixnum)) ;; (axpy n...
  (mode mode :type (and keyword (member :batch :apply :apply-flatten))))

(defstruct (WTensor
	    (:conc-name wtensor-)
	    (:constructor make-wtensor (tensor order)))
  "An wrapper of AbstractTensor not to destruct its slots"
  (tensor tensor :type AbstractTensor)
  (shape  (sync (shape tensor) order))
  (view   (sync (tensor-view tensor) order))
  (stride (sync (tensor-stride tensor) order)))

;; Making this function T under more various situations are the rational way to optimize;
(defun rest-contiguous-p (rank &rest tensors)
  "Judges if collapsing AbstractLoop for rest ranks.
If remaining loops are consisted of T or :broacast (i.e.: contiguous on memory), they're fused into 1D Tensor.
Tensors ... Able to include dynamic shape"
  (declare (type (unsigned-byte 32) rank)
	   (type list tensors))
  ;; Memory-Layout
  (flet ((not-reductable-p (tensor
			    &aux
			      (views
			       (nthcdr rank (wtensor-view tensor))))
	   
	   (some #'(lambda (v)
		     (not
		      (or (eql (force-list v) t)
			  (subscript-broadcast v))))
		 views))
	 (consistency-p (tensor
			 &aux
			   (views
			    (nthcdr rank (wtensor-view tensor))))
	   (let ((rep (viewtype (force-list (car views)))))
	     (every
	      #'(lambda (v)
		  (equal rep (viewtype (force-list v))))
	      views))))
    (and
     (not (some #'not-reductable-p tensors))
     (every #'consistency-p tensors))))
     

(defun solve-loop-order (tensors kernel-size force-order &key (mode :heuristic))
  "
Creates an optimized route of Iterations.
 mode = :heuristic or :runtime (Set :heuristic for all case)

Examples:
 kernel-size=1 ... element-wise
 kernel-size=2 ... matmul
 kernel-size=3 ... Unfold
"
  
  (declare (type list tensors)
	   (type (unsigned-byte 32) kernel-size)
	   (type boolean force-order)
	   (type (member :heuristic :runtime) mode)
	   (optimize (speed 3))
	   (ignore mode))
  
  (assert (every #'(lambda (x)
		     (>= (the fixnum (dims x)) kernel-size))
		 tensors)
	  nil
	  "Assertion Failed: Ranks are too low compared to declare: ~a" kernel-size)

  (when *freeze-call-with-view* (setq force-order t))
  (decf kernel-size)
  
  ;; Tensor(10 20 30)
  ;;            ^ when kernel-size=2
  ;; Rank=3,  ... it is ok to shuffle orders
  ;; Rank=2, 1    the order MUST BE THE SAME.

  ;; Tensor(10 20 30) kernel-size=0 (element-wise ops)
  ;; Everything is subject to be shuffled

  (let* ((tensor-rank (dims (car tensors))))
    (declare (type fixnum tensor-rank))
    (labels ((compute-loop (wtensors &optional (rank 0))
	       (declare (type fixnum rank))
	       (when (and (not force-order)
			  (= kernel-size 0)		  
			  (apply #'rest-contiguous-p rank wtensors))
		 ;; Collapsing loops because elements are contiguous on memory.
		 (let* ((iter-n
			  ;; N-Elements = MUL_UP(rank...tensor-rank)
			  (cl-waffe2/vm:make-lazyaxis
			   `(* ,@(map 'list #'(lambda (r) (find-size wtensors r)) (range rank tensor-rank)))))
			(iter-n (or
				 (cl-waffe2/vm:lazyaxis-symbol iter-n)
				 iter-n)))
		   
		   (return-from
		    compute-loop
		     (list
		      (make-aloop
		       rank 1 iter-n iter-n
		       :apply-flatten)))))

	       ;; Reached kernel-size+1 but collapsing was failed:
	       ;; -> applying CFFI function
	       (if (= (the fixnum (- tensor-rank rank)) (1+ kernel-size))
		   (let* ((list (map 'list #'(lambda (r) (find-size wtensors r)) (range rank tensor-rank)))
			  (out (cl-waffe2/vm:make-lazyaxis
				`(* ,@(loop for o in list
					    if o collect o)))))
		     (list
		      (make-aloop
		       rank
		       1
		       (or
			(cl-waffe2/vm:lazyaxis-symbol out)
			out)
		       1
		       :apply)))
		   (cons
		    (make-aloop
		     rank
		     1
		     nil
		     (find-size wtensors rank)
		     :batch)
		    (compute-loop wtensors (1+ rank))))))

      (flet ((wrapper (tensor) (make-wtensor tensor (range kernel-size tensor-rank))))
	(compute-loop (map 'list #'wrapper tensors))))))

;;#+sbcl(setf sb-ext:* inline
;; [TODO] Inline this: expand-helper
;;(declaim (inline do-compield-loop*))
(declaim (ftype (function (list function list) t) do-compiled-loop*))
(defun do-compiled-loop* (loop-blueprint function tensors
			  &aux
			    (offsets (make-array (length tensors)
						 :element-type '(unsigned-byte 32)
						 :initial-contents (map 'list #'tensor-initial-offset tensors)))
			    (max-rank (dims (car tensors))))
  (declare (type list loop-blueprint)
	   (type function function)
	   (type list tensors)
	   (type (simple-array (unsigned-byte 32) (*)) offsets)
	   (optimize (speed 3)))

  ;; ~~ [FixMe] Runtime Recomputation of strides cause reductin in performance; Delete this line: ~~~~~
  
  ;; All lazy strides should be compiled once adjust-allocation! is called.
  ;; See also: render.lisp render-tensor
  (dolist (tensor tensors)
    (when (some #'listp (tensor-stride tensor))
      (warn "do-compiled-loop*: runtime stride recomputation...")
      (setf
       (tensor-stride tensor)
       (calc-strides (translate-adjustable-shape (original-shape tensor)) (order tensor))
       (tensor-stride tensor)
       (sync (tensor-stride tensor) (reverse (tensor-permute-order tensor))))))

  (labels ((expand-helper (&optional (c 0) (offsets offsets))
	     (declare (type fixnum c)
		      (type (simple-array (unsigned-byte 32) (*)) offsets))
	     (let ((subject (nth c loop-blueprint)))
	       (when subject

		 (when (eql (aloop-mode subject) :batch)
		   (loop with rank fixnum = (aloop-rank subject)
			 for tensor in tensors
			 for position fixnum upfrom 0 do
			   (let ((start-idx
				   (if (subscript-broadcast (nth rank (tensor-view tensor)))
				       0
				       (wf/iter:range-nth
					(subscript-range (nth rank (tensor-view tensor)))
					0))))
			     (declare (type (unsigned-byte 32) start-idx))
			     (when (not (= 0 start-idx)) ;; If start-idx = 0 -> isn't worth it.
			       (incf (the fixnum (aref offsets position))
				   (the fixnum (* start-idx (the fixnum (nth rank (tensor-stride tensor))))))))))
		 
		 (if (eql (aloop-mode subject) :batch)
		     (loop with offsets = (copy-seq offsets)
			   with iter-num fixnum = (cl-waffe2/vm:maybe-observe-axis (aloop-size subject))
			   with diffs = (map
					 'list
					 #'(lambda (tensor)
					     (let ((range  (subscript-range (nth (aloop-rank subject) (tensor-view tensor))))
						   (stride (nth (aloop-rank subject) (tensor-stride tensor))))
					       (declare (type (unsigned-byte 32) stride))
					       (the
						fixnum
						(*
						 stride
						 ;; the direction range indicating:
						 (the
						  fixnum
						  (-
						   (the fixnum (wf/iter:range-nth range 1))
						   (the fixnum (wf/iter:range-nth range 0))))))))
					 tensors)
			   for index fixnum
			   upfrom 0 below iter-num do
			     (expand-helper (1+ c) offsets)
			   unless (= index (1- iter-num)) do
			     (loop for diff fixnum in diffs
				   for tensor in tensors
				   for position fixnum upfrom 0
				   unless (subscript-broadcast (nth (aloop-rank subject) (tensor-view tensor)))
				     do (incf (the fixnum (aref offsets position)) diff)))		     
		     (apply
		      function
		      (loop with rank fixnum = (aloop-rank subject)
			    with by   fixnum = (cl-waffe2/vm:maybe-observe-axis (aloop-by subject))
			    for tensor   in tensors
			    for position fixnum upfrom 0
			    collect
			    (loop with offset  fixnum = (aref offsets position)
				  for nth-rank fixnum upfrom rank below max-rank
				  for stride   fixnum in (nthcdr rank (tensor-stride tensor))
				  collect
				  (make-viewinstruction ;; (OFFSET, SIZE, STRIDE)
				   (the
				    fixnum
				    (+
				     offset ;; total offsets accumlated before :apply
				     ;; offsets created by !view
				     ;; If stride is a negative number:
				     ;; e.g.: (0 5 -1)
				     ;; the offset starts from 5
				     (unless (subscript-broadcast (nth nth-rank (tensor-view tensor)))
				       (wf/iter:range-nth
					(subscript-range
					 (nth
					  rank
					  (tensor-view tensor)))
					0))))
				   (if (eql (aloop-mode subject) :apply-flatten)
				       (cl-waffe2/vm:maybe-observe-axis (aloop-element-n subject))
				       (cl-waffe2/vm:maybe-observe-axis (nth nth-rank (shape tensor))))
				   (if (eql (aloop-mode subject) :apply-flatten)
				       (if (subscript-broadcast
					    (nth nth-rank (tensor-view tensor)))
					   0
					   1)
				       (if (subscript-broadcast (nth nth-rank (tensor-view tensor)))
					   0
					   (*
					    (the (signed-byte 32) (nth nth-rank (tensor-stride tensor)))
					    (the (signed-byte 32)
						 (-
						  (wf/iter:range-nth
						   (subscript-range
						    (nth nth-rank (tensor-view tensor)))
						   1)
						  (wf/iter:range-nth
						   (subscript-range
						    (nth nth-rank (tensor-view tensor)))
						   0)))))))))))))))					       
    ;; #+sbcl(declare (inline expand-helper))
    (expand-helper 0)
    nil))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *compiled-loop-table* (make-hash-table)))

(defun from-compiled-loop (cache-id tensors)
  "Produces keys for caching the result of solve-loop-order"
  (let ((query (loop for tensor in tensors append
		     `(,(shape tensor)
		       ,(tensor-permute-order tensor)
		       ,(map 'list #'force-list (tensor-view tensor))
		       ,*freeze-call-with-view*))))
    (gethash query (gethash cache-id *compiled-loop-table*))))

(defun set-compiled-loop (cache-id solved tensors)
  "Given cache-id, stores the result obtained by solve-loop-order."
  (let ((query (loop for tensor in tensors append
		     `(,@(shape tensor)
		       ,@(tensor-permute-order tensor)
		       ,@(map 'list #'force-list (tensor-view tensor))
		       ,*freeze-call-with-view*))))
    (setf (gethash query (gethash cache-id *compiled-loop-table*)) solved)))

(defun maybe-solve-loop (cache-id tensors kernel-size force-order mode)
  (let ((result (from-compiled-loop cache-id tensors)))
    (or result
	(let ((solved (solve-loop-order tensors kernel-size force-order :mode mode)))
	  (set-compiled-loop cache-id solved tensors)
	  solved))))

(defmacro do-compiled-loop (tensor-list (&key (kernel-size 1) (collapse t) (mode :runtime)) (&rest views-bind) &body body &aux (cache-id (gensym "LOOP_CACHE")))
  "
## [macro] do-compiled-loop

```lisp
(do-compiled-loop tensor-list (&key (kernel-size 1) (collapse t) (mode :runtime)) (&rest views-bind) &body body)
```

Iterates the given tensors in optimized order. The behavior is the same as the `call-with-view` function in that both is intended to call a ranked matrix function with considering multidimensional offsets. This macro, however, directly placed within functions. 

### Inputs

`tensor-list[list]` an list of tensors. all of them must have the same rank and shape[> kernel-size]. Must not include adjustable shape.

`kernel-size[(unsigned-byte 32)]` Indicates the rank of operation, that is, indicates the same as `:at-least-dim` in the call-with-view function.

`collapse[boolean]` Set T to enable `Loop Collapse`. (= (not :force-order) in call-with-view)

`mode[:runtime or :heuristic]` indicates the algorithm of optimizing. `:heuristic` mode is still under experimental and not tested well. So set :runtime.

### Example

```lisp
(define-impl-op (Compare-Operation-Node :device LispTensor)
		:forward ((self tensor1 tensor2 out)
			  (let ((kernel (compare-kernel (dtype tensor1))))
			    (do-compiled-loop (list tensor1 tensor2 out) ()
				(x-view y-view o-view)
			      (funcall kernel
				       (tensor-vec tensor1)
				       (tensor-vec tensor2)
				       (tensor-vec out)				       
				       (logical-condition self)
				       (logical-true-then self)
				       (logical-false-then self)
				       (size-of x-view 0)
				       (offset-of x-view 0)
				       (offset-of y-view 0)
				       (offset-of o-view 0)
				       (stride-of x-view 0)
				       (stride-of y-view 0)
				       (stride-of o-view 0)))
			    out)))
```
"

  
  `(progn
     (when (null (gethash ',cache-id *compiled-loop-table*))
       (setf (gethash ',cache-id *compiled-loop-table*) (make-hash-table :test #'equal)))
     (do-compiled-loop*
	 (maybe-solve-loop ',cache-id ,tensor-list ,kernel-size ,(not collapse) ,mode)
       #'(lambda (,@views-bind)
	   ,@body)
       ,tensor-list)))



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

(defun estimate-cost (loops)
  (declare (type list loops))
  (apply #'* (map 'list #'aloop-size loops)))

(defun estimate-simd-size (loops)
  (aloop-element-n (car (last loops))))

(defun compute-last-stride (rank wtensor)
  (if (find 1 (nthcdr rank (wtensor-stride wtensor)))
      1
      (let ((out (apply #'gcd (nthcdr rank (wtensor-stride wtensor)))))
	(if (= out 1)
	    nil
	    out ;; out [TODO] it should return out but set as nil for now. With enough test cases, we can set as 1
	    ))))

;; Making this function T under more various situations are the rational way to optimize;
(defun rest-contiguous-p (rank &rest tensors &aux (layout (nthcdr rank (wtensor-stride (car tensors)))))
  "Judges if collapsing AbstractLoop for rest ranks.
If remaining loops are consisted of T or :broacast (i.e.: contiguous on memory), they're fused into 1D Tensor."
  (declare (type (unsigned-byte 32) rank)
	   (type list tensors))
  ;; Memory-Layout
  (flet ((not-reductable-p (tensor
			    &aux
			      (views
			       (nthcdr rank (wtensor-view tensor))))
	   (or
	    ;; T :broadcast is invaild for example;
	    ;; Possible cases are: T T T... or broadcast broadcast ...
	    ;;(not
	    ;; (every #'(lambda (v)
	    ;;		(eql (force-list v)
	    ;;		     (force-list (car views))))
	    ;;	    views))
	    (some #'(lambda (v)
		      (not (or (eql (force-list v) t)
			       (eql (force-list v) :broadcast))))
		  views))))
    ;; Remaining strides are regarded as: 1 * broadcasted_p(tensor)
    ;; When Row-Major    : We can do more; Make more elements stride=1
    ;; When Column-Major : This function works
    (and
     ;; Regarding offsets
     (not (some #'not-reductable-p tensors))

     ;; Memory Layouts are the same?
     ;; [TODO] Optimize: Moving Permuted -> Not_Permuted
     ;; [TODO] With Symbols?
     ;; GCD(strides)
     (every #'(lambda (x) (equal layout (nthcdr rank (wtensor-stride x)))) tensors)
     (let ((strides (map 'list #'(lambda (x) (compute-last-stride rank x)) tensors)))
       (every #'(lambda (x) (eql (car strides) x)) strides)))))

(defun solve-loop-order (tensors kernel-size force-order &key (mode :heuristic))
  "
Creates an optimized route of AbstractLoop.
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
	   (optimize (speed 3)))

  ;;(dolist (tensor tensors)
  ;;  (when (some #'symbolp (shape tensor))
    ;;  (setf (slot-value tensor 'visible-shape) (translate-adjustable-shape (shape tensor)))))
  
  (when (not (eql mode :runtime))
    (assert (every #'(lambda (x)
		       (and
			(not (some #'symbolp (shape x)))
			(equal (butlast (shape (car tensors)) kernel-size) (butlast (shape x) kernel-size))))
		   tensors)
	    nil
	    "Assertion Failed: solve-loop-order, Tensors must be the same shape size, and not include symbols.")

    (assert (every #'(lambda (x)
		       (>= (the fixnum (dims x)) kernel-size))
		   tensors)
	    nil
	    "Assertion Failed: Ranks are too low compared to declare: ~a" kernel-size))

  (when *freeze-call-with-view*
    (setq force-order t))
  
  (decf kernel-size)
  
  ;; Tensor(10 20 30)
  ;;            ^ when kernel-size=2
  ;; Rank=3,  ... it is ok to shuffle orders
  ;; Rank=2, 1    the order MUST BE THE SAME.

  ;; Tensor(10 20 30) kernel-size=0 (element-wise ops)
  ;; Everything is subject to be shuffled

  (let* ((tensor-rank (dims (car tensors)))
	 (permutations)
	 (candidates))
    (declare (type fixnum tensor-rank))

    (labels ((all-permutations (xs &optional a)
	       (if (null xs)
		   (push (reverse a) permutations))
	       (dolist (x xs)
		 (all-permutations (remove x xs) (cons x a))))
	     (compute-loop (wtensors &optional (rank 0))
	       (declare (type fixnum rank))
	       (when (and (not force-order)
			  (= kernel-size 0)		  
			  (apply #'rest-contiguous-p rank wtensors))
		 (return-from
		  compute-loop
		   (list
		    (make-aloop
		     rank
		     (the fixnum (compute-last-stride (the fixnum rank) (car wtensors)))
		     (apply #'l* (map 'list #'(lambda (r) (find-size wtensors r)) (range rank tensor-rank)))
		     1
		     :apply-flatten))))

	       ;; Reached kernel-size+1 -> apply
	       (if (= (the fixnum (- tensor-rank rank)) (1+ kernel-size))
		   (list (make-aloop
			  rank
			  1
			  (let ((out (map 'list #'(lambda (r) (find-size wtensors r)) (range rank tensor-rank))))
			    (apply #'l* (loop for o in out if o collect o)))
			  1;;(find-size wtensors rank)
			  :apply))
		   (cons
		    (make-aloop
		     rank
		     1
		     nil
		     (find-size wtensors rank)
		     :batch)
		    (compute-loop wtensors (1+ rank))))))

      (case mode
	(:heuristic
	 (all-permutations (range kernel-size tensor-rank)))
	(:runtime
	 (push (range kernel-size tensor-rank) permutations)))

      (dolist (~ permutations)
	;; ~ ... (nth (car ~) shape), (nth (second ~) shape) ...
	(flet ((wrapper (tensor) (make-wtensor tensor ~)))
	  (push (compute-loop (map 'list #'wrapper tensors)) candidates)))

      (when (eql mode :runtime)
	;; Early Return
	(return-from solve-loop-order (car candidates)))

      ;; Sort by Costs
      (setq candidates (sort candidates #'< :key #'estimate-cost))
      ;; Candidates with the minimum costs
      (setq candidates (loop with target fixnum = (estimate-cost (car candidates))
			     for c in candidates
			     if (= (the fixnum (estimate-cost c)) target)
			       collect c))
      (setq candidates (sort candidates #'> :key #'estimate-simd-size))

      (setq candidates (loop with target fixnum = (estimate-simd-size (car candidates))
			     for c in candidates
			     if (= (the fixnum (estimate-simd-size c)) target)
			       collect c))

      (setq candidates (sort candidates #'< :key #'length))
      (car candidates))))

;;#+sbcl(setf sb-ext:* inline
;; [TODO] Inline this: expand-helper
;;(declaim (inline do-compield-loop*))
(declaim (ftype (function (list function list) t) do-compiled-loop*))
(defun do-compiled-loop* (loop-blueprint function tensors
			  &aux
			    (offsets (make-array (length tensors)
						 :element-type '(unsigned-byte 32)
						 :initial-element 0))
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
		 (loop with rank fixnum = (aloop-rank subject)
		       for tensor in tensors
		       for position fixnum upfrom 0 do
			 (let ((start-idx (compute-visible-start-idx (force-list (nth rank (tensor-view tensor))))))
			   (declare (type (unsigned-byte 32) start-idx))
			   (when (not (= 0 start-idx))
			     (incf (the fixnum (aref offsets position))
				   (the fixnum (* start-idx (the fixnum (nth rank (tensor-stride tensor)))))))))
		 
		 (if (eql (aloop-mode subject) :batch)
		     (loop with offsets = (copy-seq offsets)
			   for index fixnum
			   upfrom 0 below (aloop-size subject) do
			     (expand-helper (1+ c) offsets)
			   unless (= index (1- (aloop-size subject)))
			     do (loop for tensor in tensors
				      for position fixnum upfrom 0
				      unless (= 0 (compute-stepby
						   (force-list (nth (aloop-rank subject) (tensor-view tensor)))))
					do (incf (the fixnum (aref offsets position))
						 (the fixnum (nth (aloop-rank subject) (tensor-stride tensor))))))
		     (apply
		      function
		      (loop with rank fixnum = (aloop-rank subject)
			    with by   fixnum = (aloop-by subject)
			    for tensor   in tensors
			    for position fixnum upfrom 0
			    collect
			    (loop with offset fixnum = (aref offsets position)
				  for nth-rank fixnum upfrom rank below max-rank
				  for stride fixnum in (nthcdr rank (tensor-stride tensor))
				  collect
				  (make-viewinstruction
				   (the fixnum (+ offset (the fixnum (compute-visible-start-idx (force-list (nth rank (tensor-view tensor)))))))
				   (if (eql (aloop-mode subject) :apply-flatten)
				       (aloop-element-n subject)
				       (nth nth-rank (shape tensor)))
				   (if (eql (aloop-mode subject) :apply-flatten)
				       by
				       (compute-stepby (force-list (nth nth-rank (tensor-view tensor))))))))))))))
    ;; #+sbcl(declare (inline expand-helper))
    (expand-helper 0)
    nil))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *compiled-loop-table* (make-hash-table)))

(defun from-compiled-loop (cache-id tensors)
  (let ((query (loop for tensor in tensors append
		     `(,(shape tensor)
		       ,(tensor-permute-order tensor)
		       ,(map 'list #'force-list (tensor-view tensor))
		       ,*freeze-call-with-view*))))
    (gethash query (gethash cache-id *compiled-loop-table*))))

(defun set-compiled-loop (cache-id solved tensors)
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

;; :mode=:heuristic is still under experimental
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


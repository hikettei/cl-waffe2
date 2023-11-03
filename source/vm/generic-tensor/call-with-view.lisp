

(in-package :cl-waffe2/vm.generic-tensor)

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; [TODO] Integrate do-compiled-loop.lisp and call-with-view.lisp
;;        call-with-view perform by far the fastset performance while do-compiled-loop could be potentially optimized for permuted tensors.
;;        => OK. Now they're using the same algorithm
;;        => Next: Implementing Polyhedral Compiler
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; Force-Order=T Anywhere
(defparameter *freeze-call-with-view* nil "Set this parameter T to make force-order=t everywhere. default: nil")

;; ===============================================
;; call-with-view utils
;; ===============================================


(defun update-calling-route (value)
  ;; Utils
  ;; cl-waffe2/vm.nodes can trace the result of *call-with-view-route*
  (push value cl-waffe2/vm.nodes::*call-with-view-route*))

(defun call-with-view (function
		       tensors
		       &key
			 (at-least-dim 1)
			 (force-order nil)
			 (lparallel nil)
		       &aux
			 (shape (shape (car tensors)))
			 (force-order
			  (if (or (not lparallel)
				  (= cl-waffe2/threads:*num-cores* 1))
			      force-order
			      t)))
  "
## [function] call-with-view

Inlined Loop Macro Generator for the extension to higher order foreign functions.

```lisp
(call-with-view function tensors &key (at-least-dim 1) (force-order nil) (lparallel nil))
```

The function `call-with-view` generates a lisp code calling BLAS-like function for nd-arrays with considered offsets produced by views. Plus, depending on the memory-layouts and permutation(offsets) the generated loop is also collapsed and shuffled to maximize the locality of memory and reduce the overheads produced by calling CFFI function.

In the simplest case, `call-with-view` first deploys `(loop for...)` until the rank of given tensors reaches the given `at-least-dim`. After reaching `at-least-dim`, the function places the result of calling the given `function`.

```lisp
(call-with-view
      #'(lambda (x-view)
	   `(+ 1 1))
       (list (randn `(100 100 100)))
       :at-least-dim 1)

(LET ((#:OFFSETS68091
       (MAKE-ARRAY 1 :ELEMENT-TYPE '(UNSIGNED-BYTE 64) :INITIAL-ELEMENT 0)))
  (DECLARE (TYPE (SIMPLE-ARRAY (UNSIGNED-BYTE 64) (*)) #:OFFSETS68091))
  (PROGN (+ 1 1)))
```

Here, the number of tensors corresponds with the number of arguments `function` receive. Usually, the function receives information on the view of the tensor at the corresponding position: `(size-of x-view)` to get the number of iteration, `(stride-of x-view)` to get the number of increment, and, `(offset-of x-view)` to get the offset of tensor. (Sometimes they return s-expression because the shapes of tensors are not necessary number, but symbols.)

`function [function]` should return a list which corresponds with invoking user-defined operation given views.

`tensors[a list of abstracttensor]` tensors to be called with.

`at-least-dim [fixnum]` `at-least-dim is minimum rank value required by the operation. set 1 to define `element-wise` operation, set 2 to define `gemm` for example.

`force-order[boolean]` On some conditions, `call-with-view` shuffles the order of ranks, or flattens given tensors (e.g.: `100x100` tensors is the equivalent to just `10000x1` tensor on the memory). If you want to disable this behaviour, set `force-order`=t.

`lparallel[boolean]` Set t to use lparallel. This should be denoted that under lparallel execution, the parameter `cl-waffe2/threads:*under-multi-thread*` becomes t. Use this parameter for the lowest rank operation to decide whether to parallelise.

Return: `Expanded Lisp Codes`

Note that `call-with-view` should be used at once or zero in the one `define-impl` forward. If you need twice times to call it, the general definition of `AbstractNode` should be split.

See also: `with-ranked-loop` to the more elegant wrapping macro.
"
  
  (declare
   (type function function)
   (type list tensors shape)
   (type fixnum at-least-dim))
  
  (assert (every #'(lambda (tensor) (shape-equal-list (butlast (shape tensor) at-least-dim) (butlast shape at-least-dim))) tensors)
	  nil
	  "call-with-view failed with assertion: All all tensors has the same dimensions of batch-area:
butgot ~a."
	  (map 'list #'shape tensors)) ;; ... (1)

  (when (some #'scalar-p tensors)
    (error "call-with-view: tensors must not include ScalarTensor.
  You probably called AbstractNode excepting a Matrix with ScalarTensor.
  Use the ->mat function to create matrix from scalar."))

  (when *freeze-call-with-view*
    (setq force-order t))

  (let* ((solved-loop   (solve-loop-order tensors at-least-dim force-order :mode :heuristic))
 	 (offsets-place (gensym "OFFSETS"))
	 (diffs-place   (gensym "DIFFS"))
	 (no-batch-p    (= (length solved-loop) 1)))
    (mapc
     (compose
      #'update-calling-route
      #'aloop-rank)
     solved-loop)

    (labels ((maybe-observe-axis1 (value)
	       (or (when (numberp value) value)
		   `(cl-waffe2/vm:maybe-observe-axis ',value)))
	     (expand-helper (rank)
	       (let ((subject (nth rank solved-loop)))
		 (when (null subject) (return-from expand-helper))
		 `(progn
		    ;; Initial Offsets		    
		    ,@(when (eql (aloop-mode subject) :batch)
			(loop with rank fixnum = (aloop-rank subject)
			      for tensor in tensors
			      for position upfrom 0
			      collect
			      (let ((start-idx
				      (if (subscript-broadcast (nth rank (tensor-view tensor)))
					  0
					  `(wf/iter:range-nth
					    (subscript-range ,(nth rank (tensor-view tensor)))
					    0))))
				(when (listp start-idx)
				  `(incf
				       (the fixnum (aref ,offsets-place ,position))
				       (the fixnum
					    (*
					     (the fixnum ,start-idx) 
					     ,(nth rank (tensor-stride tensor)))))))))

		    ;; Exploring remaining loops
		    ,(if (eql (aloop-mode subject) :batch)
			 (alexandria:with-gensyms (total-count count)
			   `(loop with ,offsets-place = (copy-seq ,offsets-place)
				  with ,total-count   = ,(maybe-observe-axis1 (aloop-size subject))
				  for ,count of-type fixnum upfrom 0 below ,total-count
				  do (progn
				       ,(expand-helper (1+ rank))
				       ,(when (not
					       (and
						(numberp (aloop-size subject))
						(= (aloop-size subject) 1)))
					  `(unless (= ,count (1- ,total-count))
					     (progn
					       ,@(loop with dim = (aloop-rank subject)
						       for tensor in tensors
						       for pos upfrom 0
						       unless (subscript-broadcast (nth dim (tensor-view tensor)))
							 collect
						       `(incf
							    (the fixnum (aref ,offsets-place ,pos))
							    (aref ,diffs-place ,(aloop-rank subject) ,pos)))))))))
			 (apply
			  function
			  (loop for tensor in tensors
				for position upfrom 0
				collect
				(loop with offsets  = `(the fixnum (aref ,offsets-place ,position))
				      for  nth-rank upfrom rank below (dims (car tensors))
				      collect
				      (make-viewinstruction
				       `(the (signed-byte 32)
					     (+ ,offsets
						,@(unless (subscript-broadcast (nth nth-rank (tensor-view tensor)))
						    `((the fixnum
							   (wf/iter:range-nth
							    ,(subscript-range
							      (nth nth-rank (tensor-view tensor)))
							    0))))))
				       (if (eql (aloop-mode subject) :apply-flatten)					  
					   `(the fixnum
						 ,(maybe-observe-axis1
						   (aloop-element-n subject)))
					   `(the fixnum
						 ,(maybe-observe-axis1
						   (nth nth-rank (shape tensor)))))
				       (if (eql (aloop-mode subject) :apply-flatten)
					   (if (subscript-broadcast
						(nth nth-rank (tensor-view tensor)))
					       0
					       1)
					   (if (subscript-broadcast
						(nth nth-rank (tensor-view tensor)))
					       0
					       `(the fixnum (aref ,diffs-place ,rank ,position)))))))))))))
      `(let ((,offsets-place (make-array
			      ,(length tensors)
			      :element-type 'fixnum
			      :initial-contents (list ,@(map 'list #'(lambda (tensor) `(tensor-initial-offset ,tensor)) tensors))))
	     ,@(when (not no-batch-p)
		 `((,diffs-place (make-array
				  (list ,(length solved-loop) ,(length tensors))
				  :element-type 'fixnum
				  :initial-contents
				  (list
				   ,@(loop for aloop in solved-loop
					   for n-rank upfrom 0
					   collect
					   `(list
					     ,@(loop with rank = (if (eql (aloop-mode aloop) :batch)
								     (aloop-rank aloop)
								     n-rank)
						     for tensor in tensors
						     collect
						     (let ((view (subscript-range (nth rank (tensor-view tensor)))))
						       `(the
							 fixnum
							 (*
							  ,(nth rank (tensor-stride tensor))
							  (the
							   fixnum
							   (-
							    (the
							     fixnum
							     (wf/iter:range-nth ,view 1))
							    (the
							     fixnum
							     (wf/iter:range-nth ,view 0))))))))))))))))
	 (declare (type (simple-array fixnum (*)) ,offsets-place)
		  ,@(when (not no-batch-p) `((type (simple-array fixnum (* *)) ,diffs-place) (ignorable ,diffs-place))))
	 ,(expand-helper 0)))))

(defmacro with-ranked-loop (((op-function &rest variables)
			     &key
			       (kernel-size 1)
			       (shuffle-rank t)
			       (lparallel nil))
			    &body
			      body)
  "
## [macro] with-ranked-loop


```lisp
(with-ranked-loop (((op-function &rest variables)
                    &key
                       (kernel-size 1)
                       (shuffle-rank t)
                       (lparallel nil))
                    &body body))
```

Just an alias of `call-with-view` with this form:

```lisp
`(,@(call-with-view op-function variables :at-least-dim kernel-size :force-order (not shuffle-rank) :lparallel lparallel :fuse fuse)
  ,@body)
```
"
  `(,@(call-with-view op-function variables :at-least-dim kernel-size :force-order (not shuffle-rank) :lparallel lparallel)
    ,@body))


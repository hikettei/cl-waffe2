

(in-package :cl-waffe2/vm.generic-tensor)

;; cl-waffe2 has two mode depending on the situation

;;
;; build:   Supports FuseOps/Fully Inlining (Memo: cl-waffe2 defnode corresponds with IR, conditions, iterations are expressed/implemented as AbstractNode)
;;
;; proceed: No supports of FuseOps but working enough fast.
;;

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; In cl-waffe2, call-with-view is a function used to express an iteration on an AbstractTensor.
;; And, it is intended to be used for each single operation unit (exp/sin/matmul ...)
;;
;; Taking the case of the element-wise function `exp`, the body of :forward can be expressed like:
;;

;; ====================================================
;; (loop for i <- (Index considered views)              }
;;      [Repeating for the rank of tensors]             } <- Expanded by call-with-view
;;      ...                                             }
;;      (element-wise-exp tensor stride offset size)    <- Kernel (user-defined)
;; ====================================================

;;
;; In addition, cl-waffe2 can apply these optimization methods to the coming tensors:
;;
;; 1. Loop Fusion
;;
;; A(x) = (loop for i ...
;;          (element-wise-sin ...))
;;
;; B(x) = (loop for i ...
;;          (element-wise-cos ...))
;;
;; Composing A and B (i.e.: A(B(x))), the expanded form would be like:
;;
;; (loop for i ...
;;          (element-wise-sin ...)
;;          (element-wise-cos ...))
;;
;; Here's more, `aref` is still remained to be optimized:
;;
;; -> Since loop Fusion is still hard to implement across multiple devices, and I decide to implement it as an extended device, JITLispTensor.
;;
;;

;; 2. Inlining
;;    If the ranks/dimensions are enough small and (LOOP COST) >> (Computation_Time), they're inlined:
;;
;; 3. Disregarding Views
;;
;;    (10 10) Tensor with view = (T T) -> (100) Tensor as long as kernel-size = 1
;;
;;
;; 4. Parallelize (TODO)
;;
;;
;;

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;;
;; call-with-view is able to generate:
;;   1. Inlined/Optimized/Parallelized Orders with coming tensors ( *freeze-call-with-view*=nil )
;;   2. Flexible Loop Iterations for NDArray.                     ( *freeze-call-with-view*=t   )
;;

(defparameter *freeze-call-with-view* nil "If this parameter set to T, the generated call-with-view is not inlined but compatible with ND-Array.
Nodes generated with this parameter=t, will forcibly compiled, never cached.")

;; ===============================================
;; call-with-view utils
;; ===============================================

(defun tensor-gensym-list (tensors) (loop for tensor in tensors collect (gensym)))

(defun expand-first-offset-adder (tensors
				  offset-places
				  stride-places
				  start-points)
  (loop for k upfrom 0
	for tensor in tensors
	collect
	;; offset += stride * start-points
	`(incf ,(nth k offset-places)
	       (%* ,(nth k start-points)
		   ,(nth k stride-places)))))

(defun expand-view-stride-adder (offset-places
				 stride-places
				 target-dim
				 tensors)
  (loop for k fixnum upfrom 0
	for tensor in tensors
	collect (let* ((view (subscript-view (nth target-dim (tensor-view tensor))))
		       (viewtype (viewtype view)))
		  (cond
		    ((or (eql viewtype :index)
			 (eql viewtype :broadcast))
		     ;; when Tensor[Index], iternum = 1 therefore there's no need to incr offsets.
		     ;; when :broadcast, freeze the axis stride.
		     nil)
		    ((or (eql viewtype :t)
			 (eql viewtype :slice))
		     `(incf (the fixnum ,(nth k offset-places))
			    (the fixnum ,(nth k stride-places))))
		    ((eql viewtype :slice-step)
		     `(incf (the fixnum ,(nth k offset-places))
			    (%* ,(third view)
				(the fixnum ,(nth k stride-places)))))
		    ((eql viewtype :indices)
		     (error ":INDICES IS NOT IMPLEMENTED"))
		    ((eql viewtype :tflist)
		     (error ":TFLIST IS NOT IMPLEMENTED"))
		    (T
		     (error "Unknown keyword ~a" viewtype))))))

(defun compute-index (offset stride target-dim ith tensor)
  (let* ((view     (subscript-view (nth target-dim (tensor-view tensor))))
	 (viewtype (viewtype view)))
    (cond
      ((or (eql viewtype :index)
	   (eql viewtype :broadcast))
       offset)
      ((or (eql viewtype :t)
	   (eql viewtype :slice))
       `(+ (the fixnum ,offset) (%* ,stride ,ith)))
      ((eql viewtype :slice-step)
       `(+ (the fixnum ,offset) (%* ,ith (%* ,(third view) (the fixnum ,stride)))))
      (T (error "Unsupported view keyword: ~a" viewtype)))))

;; e.g: (view tensor `(0 2) t t) could be splitted into: `(0 2) * t*t times order
(defun order-reductable-p (dim-start-from &rest tensors)
  "
Returns T if the rest axes of all tensors, has no views.
dim-start-from:
Set 1 if the operation is element-wise
Set 2 if the operation is matmul for example.

`(t t t) -> t"
  (flet ((not-reductable-p (tensor
			    &aux
			      (views
			       (nthcdr dim-start-from (tensor-view tensor))))
	   (or (scalar-p tensor) ;; tensor is scalar
	       ;; at least one of (nthcdr dim-start-from (tensor-view tensor)) isn't T
	       (some #'(lambda (v)
			 ;; non-reductable dim is: NOT(T) or NOT (:BROADCAST)
			 (or (not (eql (force-list v) t))
			     ;;(not (eql (force-list v) :broadcast))
			     ))
		     views))))
    ;; If tensors are consisted of non-projected-tensor...?
    (not (some #'not-reductable-p tensors))))


(defun expand-funcall-with-view (function tensors offsets-place target-dim rest-dims)
  ""
  ;; (apply function view1 view2 view3 ...)

  (apply function
	 (loop for kth-tensor upfrom 0
	       for tensor in tensors
	       collect
	       ;; Iterate for kernel-dim
	       (loop for target-dim-n
		     upfrom target-dim
		       below (+ rest-dims target-dim)
		     collect (make-viewinstruction
			      (nth kth-tensor offsets-place)
			      (if (symbolp (nth target-dim-n (shape tensor)))
				  `(read-adjustable-symbol (nth ,target-dim-n (shape ,tensor)))
				  (nth target-dim-n (shape tensor)))
			      (let ((stride (nth target-dim-n  (tensor-stride tensor)))
				    (view   (subscript-view (nth target-dim-n (tensor-view tensor)))))
				(lazy* stride (compute-stepby view))))))))

(defun expand-call-with-view-flatten
    (function
     tensors
     offset-place
     target-dim
     &key
       (dim-start-from 0))
  ;; At-least-dim = 1
  (let* ((size-list (mapcar
		     #'(lambda (tensor
				&aux				  
				  (s (shape tensor))
				  (v (tensor-view tensor)))
			 (loop for i upfrom dim-start-from below (length s)
			       unless (eql (force-list (nth i v)) t)
				 do (error "Internal Error: call-with-view-1dkernel is only applied to view=t axes.")
			       if (symbolp (nth i (shape tensor)))				 
				 collect `(read-symbol (nth ,i (shape ,tensor)))
			       else
				 collect (nth i (shape tensor))))
		     tensors))
	 (sizes (map 'list #'(lambda (x) (apply #'lazy-mulup x)) size-list)))

    ;; sizes (for exmaple) = ((100) (100))
    ;; for element-wise operation, whenever row/column major, set stride=1


    ;; (THE FIXNUM (* (THE FIXNUM 3) (THE FIXNUM A))) ..

    (apply
     function
     (loop for tensor in tensors
	   for k upfrom 0
	   collect (let ((view (make-viewinstruction
				(nth k offset-place)
				(if (symbolp (nth k sizes))
				    `(read-adjustable-symbol ,(nth k sizes))
				    (nth k sizes))
				(compute-stepby
				 (subscript-view (nth target-dim (tensor-view tensor)))))))
		     (list view))))))


(defmacro with-expand-init-tmp-form (offset-name-place tensors &body body)
  "Expands: initializing offsets, determining symbols form

Return: (values offsets-place form)"

  `(let ((,offset-name-place (tensor-gensym-list ,tensors)))
     ;; Initializing Offsets with 0
     `(let*-ignorable (,@(loop for name in ,offset-name-place
			       for tensor in tensors
			       collect `(,name (tensor-initial-offset ,tensor))))
	(locally (declare (type fixnum ,@,offset-name-place))
	  ,,@body))))


(defmacro with-update-offset-place (offset-name-place tensors &body body &aux (tmp-space (gensym)))
  ""
  `(let ((,tmp-space ,offset-name-place)
	 (,offset-name-place (tensor-gensym-list ,tensors)))
     `(let*-ignorable (,@(loop for name in ,offset-name-place
			       for past-value in ,tmp-space
			       collect `(,name ,past-value)))
	(locally (declare (type fixnum ,@,offset-name-place))
	  ,,@body))))

(defparameter *under-lparallel* nil)

;; TODO: SIMDfied mathematical functions, inline view offsets.
(defmacro with-expanding-explore-form ((lparallel tensors offset-places target-dim start-points end-points) &body body &aux (endpoint-place (gensym)))
  ;; Set Strides At Runtime
  ;; Expand Loop      
  `(let ((stride-places (tensor-gensym-list ,tensors))
	 (ith (gensym)))
     `(let* (,@(loop for stride-place in stride-places ;; (place <- stride)
		     for tensor in ,tensors
		     collect `(,stride-place (nth ,,target-dim (list ,@(tensor-stride tensor)))))
	     (,',endpoint-place ,(car ,end-points))
	     (,',endpoint-place (read-adjustable-symbol ,',endpoint-place)))

	,@(expand-first-offset-adder
	   ,tensors
	   ,offset-places
	   stride-places
	   ,start-points)

	,(if (and (not *under-lparallel*) ,lparallel (not (= 1 cl-waffe2/threads:*num-cores*)))
	     (let ((*under-lparallel* t))
	       `(cl-waffe2/threads:maybe-pdotimes (,ith
						   ,',endpoint-place
						   :thread-safe-vars ,,offset-places
						   :disable-p (<= (apply #'* (translate-adjustable-shape (shape ,(car ,tensors))))
								  cl-waffe2/threads:*multithread-threshold*))
		  (let* (,@(loop for offset in ,offset-places
				 for k fixnum upfrom 0
				 for tensor in tensors
				 collect
				 `(,offset ,(compute-index offset (nth k stride-places) ,target-dim ith tensor))))
		    ,,@body)))
	     ;; Expand Multi-Dimensional Looping Forms
	     `(loop for ,ith fixnum upfrom 0 below ,',endpoint-place
		    ;; 1. Execute Operation
		    ;; 2. Adding Offsets
		    do (progn ,,@body)
		    unless (= ,ith (1- ,',endpoint-place))
		      ;; Unless islast, expand it.
		      do (progn ,@(expand-view-stride-adder ,offset-places stride-places ,target-dim ,tensors)))))))

(defun update-calling-route (value)
  (push value cl-waffe2/vm.nodes::*call-with-view-route*))

(defmacro with-bind-shape (&body body)
  `(flet ((original-shape (tensor)
	    (translate-adjustable-shape (original-shape tensor)))
	  (shape (tensor)
	    (translate-adjustable-shape (shape tensor))))
     ,@body))

(defun no-permute-p (tensors)
  (flet ((check (tensor)
	   (let ((k (loop for i upfrom 0 below (length (shape tensor))
			  collect i)))
	     ;; Permute Order is 3 2 1...?
	     (equal (reverse k) (tensor-permute-order tensor)))))
    (every #'check tensors)))

(defmacro cl-waffe2-internal-tagbody (tag &body body)
  "(cl-waffe2-internal-tagbody #:TAG000
     ... body)

Used to replace/fuse generated call-with-view."
  (declare (ignore tag))
  `(progn ,@body))


;; Using Ranked-Loop Information at compiling time, and later expand into Fused Operation:
;; The step seems ugly:
;; After call-with-view is called, the structure Ranked-Loop is set to *ranked-loop-result-cacher*
(defparameter *ranked-loop-result-cacher* nil)
;; define-impl macro binds this parameter, and then reading this variable.

;; So, one define-impl :forward, call-with-view can be used at once.

(defstruct (Ranked-Loop
	    (:conc-name rloop-))
  (expanded-body nil :type list)
  (op-function nil :type function)
  (tensors nil :type list)
  (n-tensors 0 :type fixnum)
  (kernel-size 0 :type fixnum)
  (force-order nil :type boolean)
  (view-route nil :type list)
  (fuse-p     nil :type boolean)
  (lparallel  nil :type boolean)
  (tagid      nil :type symbol))

(defun place-inside-of (old-loop old-body new-body)
  (let ((target-id (rloop-tagID old-loop)))
    (labels ((fuse-op-helper (fn tree)
	       ;; "return (values XXX t) to stop exploring"
	       (multiple-value-bind (tree stop-p) (funcall fn tree)
		 (if (and (listp tree) (not stop-p))
		     (mapcar (lambda (subtree)
			       (fuse-op-helper fn subtree))
			     tree)
		     tree))))
      (fuse-op-helper
       #'(lambda (tree)
	   (if (listp tree)
	       (if (and
		    (eql (car tree) 'cl-waffe2-internal-tagbody)
		    (eql (second tree) target-id))
		   (progn
		     ;;(print tree)
		     ;;(print new-body)
		     (values
		      (append
		       new-body
		       (cdddr tree))
		      nil))
		   tree)
	       tree))
       old-body))))

(defun fuse-generated-iteration (out old-ranked-loop new-ranked-loop old-body &key (merge-body nil) (merge-loop nil))
  "Replaces the body of (c-waffe2-internal-tagbody ...) with new, fused body

Based on the information from old-ranked-loop and new-ranked-loop, this funciton replaces the old-ranked-loop.tagID body in old-body with the body of new-ranked-loop"

  (let ((target-id (rloop-tagID old-ranked-loop)))
    (labels ((fuse-op-helper (fn tree)
	       ;; "return (values XXX t) to stop exploring"
	       (multiple-value-bind (tree stop-p) (funcall fn tree)
		 (if (and (listp tree) (not stop-p))
		     (mapcar (lambda (subtree)
			       (fuse-op-helper fn subtree))
			     tree)
		     tree))))
      (fuse-op-helper
       #'(lambda (tree)
	   ;; Finds out this part: (cl-waffe2-internal-tagbody tag &body ...)
	   (if (listp tree)
	       (if (and
		    (eql (car tree) 'cl-waffe2-internal-tagbody)
		    (eql (second tree) target-id))
		   (values
		    ;; (cl-waffe2-internal-tagbody ID ...)
		    (append		  
		     (if merge-loop
			 (place-inside-of
			  merge-loop
			  merge-body
			  (rloop-expanded-body new-ranked-loop))
			 (rloop-expanded-body new-ranked-loop))
		     (cdddr tree))
		    t)
		   tree)
	       tree))
       old-body))))

;; The function it. composes the given two ranked-loop if possible, otherwise return nil
(defun it.-able-p (ranked-loop1 ranked-loop2)
  "
## [function] it.-able-p

```lisp
(it.-able-p ranked-loop1 ranked-loop2)
```

Returns t If the two given Ranked-Loops are composable otherwise nil.

Composable Ranked-Loop is defined as:

1. the value of `lparallel`, `view-route`, `force-order`, and `kernel-size` corresponds with.

2. Both of `fuse-p` is t

3. The number of `tensors` do match.

"
  (declare (type (or null Ranked-Loop) ranked-loop1 ranked-loop2))
  (and

   ;; define-impl is for Tensors?
   (not (null ranked-loop1))
   (not (null ranked-loop2))

   (rloop-fuse-p ranked-loop1)
   (rloop-fuse-p ranked-loop2)

   (eql (rloop-lparallel ranked-loop1)
	(rloop-lparallel ranked-loop2))

   (eql (rloop-kernel-size ranked-loop1)
	(rloop-kernel-size ranked-loop2))

   (eql (rloop-force-order ranked-loop1)
	(rloop-force-order ranked-loop2))

   ;; Applies to the same size?
   (let ((rep (shape (car (rloop-tensors ranked-loop1)))))
     (and (every #'(lambda (s) (equal (shape s) rep))
		 (rloop-tensors ranked-loop1))
	  (every #'(lambda (s) (equal (shape s) rep))
		 (rloop-tensors ranked-loop2))
	  ))
   
   ;; Sort by Ranks, instead of view-route? to fuse sum
   ;; (equal (rloop-view-route ranked-loop1)
   ;;	    (rloop-view-route ranked-loop2))
   ))

;; TODO: Shuffling Views to compose more operators
;; TODO: how do we handle with with-tensor-ptr?
;; We should embedding codes at a cetrain position
;; Gensym: a-ptr etc..
(defun it. (ranked-loop1 ranked-loop2 &key (embedding nil))
  "
## [function] it.

`it.` is the operator to compose ranked-loop1 and ranked-loop2 if possible.

Return: brand new composed Ranked-Loop
"
  (declare (type (or null Ranked-Loop) ranked-loop1 ranked-loop2))

  (assert (it.-able-p ranked-loop1 ranked-loop2)
	  nil
	  "it.: Assertion Failed because the given two Iterations aren't composable.")

  (let ((*ranked-loop-result-cacher*)
	(tensors `(,@(rloop-tensors ranked-loop1)
		   ,@(rloop-tensors ranked-loop2))))
    (call-with-view
     #'(lambda (&rest views)
	 (let* ((argn1  (- (length views) (rloop-n-tensors ranked-loop1)))
		(views1 (butlast views argn1))
		(views2 (last    views argn1)))
	   `(progn
	      ,(apply (rloop-op-function ranked-loop1) views1)
	      ,embedding
	      ,(apply (rloop-op-function ranked-loop2) views2))))
     tensors
     :at-least-dim (rloop-kernel-size ranked-loop1)
     :force-order  (or (rloop-force-order ranked-loop1) (rloop-force-order ranked-loop2))
     :lparallel    (rloop-lparallel   ranked-loop1)
     :fuse t)
    *ranked-loop-result-cacher*))

;; TODO: Ranked-Loop . Ranked-Loop -> New-Ranked-Loop

(defun call-with-view (function
		       tensors
		       &key
			 (at-least-dim 1)
			 (force-order nil)
			 (lparallel nil)
			 (fuse nil)
		       &aux
			 (shape (shape (car tensors)))
			 (dims  (length shape))
			 (force-order (if (or (not lparallel)
					      (= cl-waffe2/threads:*num-cores* 1))
					  force-order
					  t)))
  "
## [function] call-with-view

A principle operator to extend your functions to higher arrays.

```lisp
(call-with-view function tensors &key (at-least-dim 1) (force-order nil) (lparallel nil) (fuse nil))
```

The function `call-with-view` generates a lisp code of `(loop for ...)` iteration for nd-arrays, which follows the optimal route, is parallelized, and later composable. Since generating an optimal `for(int i=0;i<size;i++){...}` route according to the given rank of tensors is one of the main concerns of JIT Compiler for Deep Learning Framework, this function is usually combined with the forward definition of `define-impl` macro. It is later compiled to lambda functions and used as nodes in cl-waffe2 IR.

In the simplest case, `call-with-view` first deploys `(loop for...)` until the rank of given tensors reaches the given `at-least-dim`. After reaching `at-least-dim`, the function places the result of calling the given `function`.

```lisp
(call-with-view
      #'(lambda (x-view)
	   `(+ 1 1))
       (list (randn `(100 100 100)))
       :at-least-dim 2)

;; will return:

(CL-WAFFE2/VM.GENERIC-TENSOR::LET*-IGNORABLE ((#:G312057 0))
  (LOCALLY
   (DECLARE (TYPE FIXNUM #:G312057))
   (CL-WAFFE2/VM.GENERIC-TENSOR::LET*-IGNORABLE ((#:G312058 #:G312057))
     (LOCALLY
      (DECLARE (TYPE FIXNUM #:G312058))
      (LET* ((#:G312059 (NTH 0 (LIST 10000 100 1)))
             (#:G25 100)
             (#:G25
              (CL-WAFFE2/VM.GENERIC-TENSOR::READ-ADJUSTABLE-SYMBOL #:G25)))
        (INCF #:G312058 (CL-WAFFE2/VM.GENERIC-TENSOR::%* 0 #:G312059))
        (LOOP CL-WAFFE2/VM.GENERIC-TENSOR::FOR #:G312060 FIXNUM CL-WAFFE2/VM.GENERIC-TENSOR::UPFROM 0 CL-WAFFE2/VM.GENERIC-TENSOR::BELOW #:G25
              DO (PROGN
                  (CL-WAFFE2/VM.GENERIC-TENSOR::LET*-IGNORABLE ((#:G312061
                                                                 #:G312058))
                    (LOCALLY
                     (DECLARE (TYPE FIXNUM #:G312061))
                     (LET ((#:G312062 (THE FIXNUM (NTH 1 (LIST 10000 100 1)))))
                       (INCF #:G312061
                             (CL-WAFFE2/VM.GENERIC-TENSOR::%* 0 #:G312062))
                       (+ 1 1)))))
              UNLESS (= #:G312060 (1- #:G25))
              DO (PROGN
                  (INCF (THE FIXNUM #:G312058) (THE FIXNUM #:G312059)))))))))
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
  
  (declare ;;(optimize (speed 3))
   (type function function)
   (type list tensors shape)
   (type fixnum at-least-dim dims))
  
  (assert (every #'(lambda (tensor) (shape-equal-list (butlast (shape tensor) at-least-dim) (butlast shape at-least-dim))) tensors)
	  nil
	  "call-with-view failed with assertion: All all tensors has the same dimensions of batch-area:
butgot ~a."
	  (map 'list #'shape tensors)) ;; ... (1)

  ;; If this parameter=t, call-with-view never generate inlined iteration but iteration for ND-array
  ;; Instead, using function
  (when *freeze-call-with-view*
    (return-from call-with-view
      (expand-call-with-view* function tensors at-least-dim)))

  (labels ((explore (rest-dim offsets-place &aux (target-dim (- dims rest-dim)))
	     (declare (type fixnum rest-dim target-dim)
		      (type list offsets-place))
	     ;; Exploring ND .. 3D 2D 1D

	     ;; When The Rest Form Can be flatten
	     (when (and (= at-least-dim 1) ;; Element-Wise Operation
			(not force-order)
			(no-permute-p tensors)
			(apply #'order-reductable-p target-dim tensors) ;; check views
			(not (= rest-dim 0))) ;; If rest-dim = 0, use normal ver.
	       
	       ;; Register the route as FLATTEN
	       (update-calling-route nil)
	       (return-from explore
		 (expand-call-with-view-flatten
		  function
		  tensors
		  offsets-place
		  target-dim
		  :dim-start-from target-dim)))

	     ;; Register route as Nth-dim
	     (update-calling-route rest-dim)
	     ;; Otherwise...

	     ;; Computing Multi-Dimensional Offsets
	     (let* ((start-points (loop for tensor in tensors
					collect
					;; Here, should be computed in advance to reduce the size of compiled code.
					`(compute-visible-start-idx
					  (subscript-view (nth ,target-dim (tensor-view ,tensor))))))
		    (end-points (loop for tensor in tensors
				      collect
				      `(compute-visible-end-idx
					(subscript-view (nth ,target-dim (tensor-view ,tensor)))
					(nth ,target-dim (original-shape ,tensor))))))
	       (cond
		 ((<= rest-dim at-least-dim)

		  ;; If elements are contiguous in memory, flatten as 1D vector.
		  ;; Otherwise do element-wise?
		  (with-update-offset-place offsets-place tensors
		    (let ((stride-places (tensor-gensym-list tensors)))
		      `(let (,@(loop for stride-place in stride-places
				     for tensor in tensors
				     collect `(,stride-place (the fixnum (nth ,target-dim (list ,@(tensor-stride tensor)))))))
			 ,@(expand-first-offset-adder
			    tensors
			    offsets-place
			    stride-places
			    start-points)
			 ,(expand-funcall-with-view
			   function
			   tensors
			   offsets-place
			   target-dim
			   rest-dim)))))
		 (T
		  ;; batching
		  (with-update-offset-place offsets-place tensors
		    (with-expanding-explore-form
			(lparallel tensors offsets-place target-dim start-points end-points)
		      (explore
		       (1- rest-dim)
		       offsets-place))))))))

    (let* ((tag (gensym "INTERNAL-TAG"))
	   (result
	     (with-expand-init-tmp-form offset-place tensors
	       (explore dims offset-place)))
	   (result `(cl-waffe2-internal-tagbody ,tag ,result)))

      (setf *ranked-loop-result-cacher*
	    (make-ranked-loop
	     :expanded-body result
	     :op-function function
	     :tensors tensors
	     :n-tensors (length tensors)
	     :force-order force-order
	     :kernel-size at-least-dim
	     ;;:view-route (copy-list cl-waffe2/vm.nodes::*call-with-view-route*)
	     :fuse-p fuse
	     :lparallel lparallel
	     :tagID tag))
      result)))


(defmacro with-ranked-loop (((op-function &rest variables)
			     &key
			       (kernel-size 1)
			       (shuffle-rank t)
			       (lparallel nil)
			       (fuse nil))
			    &body
			      body)
  "
## [macro] with-ranked-loop


```lisp
(with-ranked-loop (((op-function &rest variables)
                    &key
                       (kernel-size 1)
                       (shuffle-rank t)
                       (lparallel nil)
                       (fuse nil))
                    &body body))
```

Just an alias of `call-with-view` with this form:

```lisp
`(,@(call-with-view op-function variables :at-least-dim kernel-size :force-order (not shuffle-rank) :lparallel lparallel :fuse fuse)
  ,@body)
```
"
  `(,@(call-with-view op-function variables :at-least-dim kernel-size :force-order (not shuffle-rank) :lparallel lparallel :fuse fuse)
    ,@body))

(defun expand-call-with-view* (function tensors at-least-dim)
  (let* ((offsets (loop for tensor in tensors collect (gensym "offset")))
	 (strides (loop for tensor in tensors
			collect (loop for rank upfrom 0 below at-least-dim
				      collect (gensym "strides"))))
	 (sizes   (loop for tensor in tensors
			collect (loop for rank upfrom 0 below at-least-dim
				      collect (gensym "size"))))

	 (views   (loop for tensor in tensors
			for offset in offsets
			for stride in strides
			for size   in sizes
			collect
			(loop for rank upfrom 0 below at-least-dim
			      collect
			      (make-viewinstruction offset (nth rank size) (nth rank stride)))))
	 (strides (alexandria:flatten strides))
	 (sizes   (alexandria:flatten sizes))
	 
	 (kernel-function (apply function views))
	 (kernel-applier  `(lambda (,@offsets ,@strides ,@sizes)
			     (declare (ignorable ,@offsets ,@strides ,@sizes)
				      (type fixnum ,@offsets ,@strides ,@sizes))
			     ,kernel-function)))
    ;; kernel-function ... (blas-sadd ... tensor1 tensor2 offsetXXX sizeXXX ...)
    
    `(with-bind-shape
       #'original-shape
       #'shape
       (call-with-view-function*
	(list ,@tensors)
	,at-least-dim
	,kernel-applier))))

;; [TODO] Use lparallel
(declaim (inline call-with-view-function*))
(defun call-with-view-function* (tensors at-least-dim kernel-applier)
  (declare (type list tensors)
	   (type fixnum at-least-dim)
	   (type function kernel-applier)
	   (optimize (speed 3)))

  (let* ((total-offsets (loop for tensor in tensors
			      collect (tensor-initial-offset tensor)))
	 (apply-strides (loop for tensor in tensors
			      collect (loop for axis upfrom 0 below at-least-dim
					    collect (nth axis (tensor-stride tensor)))))
	 (apply-sizes   (loop for tensor in tensors
			      collect (loop for axis upfrom 0 below at-least-dim
					    collect (nth at-least-dim (shape tensor)))))
	 (rank (the fixnum (dims (car tensors))))
	 (all-iternums
	   (loop for axis fixnum downfrom (1- rank) to at-least-dim
		 collect
		 (loop for tensor in tensors
		       collect (nth axis (shape tensor)))))
	 (all-strides
	   (loop for axis fixnum downfrom (1- rank) to at-least-dim
		 collect
		 (loop for tensor in tensors
		       collect (nth axis (tensor-stride tensor)))))
	 (start-points 
	   (loop for axis fixnum downfrom (1- rank) to at-least-dim
		 collect
		 (loop for tensor in tensors
		       collect (the fixnum (compute-visible-start-idx (force-list (nth axis (tensor-view tensor))))))))
	 (end-points nil))
;;	   (loop for axis fixnum downfrom (1- rank) to at-least-dim
;;		 collect (loop for tensor in tensors
;;			       collect (compute-visible-end-idx (nth axis (tensor-view tensor)) (nth axis (shape tensor)))))))
    
    (labels ((expand-helper (rest-dim more-iternums more-strides more-start-offsets more-end-offsets)
	       (declare (type fixnum rest-dim)
			(type list more-iternums more-strides more-start-offsets more-end-offsets))
	       (cond
		 ((<= rest-dim at-least-dim)
		  ;; call the operation
		  (apply kernel-applier
			 ;; ,@offsets ,@strides ,@sizes
			 `(,@total-offsets
			   ,@(alexandria:flatten apply-strides)
			   ,@(alexandria:flatten apply-sizes))))
		 (T

		  ;; Adding first offsets
		  (loop for stride        fixnum in (car more-strides)
			for start-offsets fixnum in (car more-start-offsets)
			for nth           fixnum    upfrom 0
			do (incf (the fixnum (nth nth total-offsets))
				 (the fixnum (* start-offsets stride))))
		  
		  (loop with last-idx fixnum = (1- (the fixnum (caar more-iternums)))
			for N fixnum upfrom 0 below (caar more-iternums) do
			  (expand-helper (1- rest-dim) (cdr more-iternums) (cdr more-strides) (cdr more-start-offsets) (cdr more-end-offsets))
			unless (= N last-idx)
			  do (loop for stride fixnum in (car more-strides)
				   for nth    fixnum upfrom 0
				   do (incf (the fixnum (nth nth total-offsets)) stride)))))))
      
      
      (assert (every #'(lambda (x) (equal (dims x) (dims (car tensors)))) tensors)
	  nil
	  "call-with-view-function*: Assertion Failed because all tensors should have the same rank but they don't.
~a" (map 'list #'shape tensors))

      (expand-helper rank all-iternums all-strides start-points end-points)
      nil)))


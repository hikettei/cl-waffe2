

(in-package :cl-waffe2/vm.generic-tensor)


;; ===============================================
;; call-with-view utils
;; ===============================================

(defun tensor-gensym-list (tensors)
  (loop for tensor in tensors
	collect (gensym)))

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
		     ;; when :broadcast, freeze the axis.
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
			 (not (eql (force-list v) t)))
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
	       (loop for target-dim
		     upfrom target-dim
		       below (+ rest-dims target-dim)
		     collect (make-viewinstruction
			      (nth kth-tensor offsets-place)
			      (nth target-dim (shape tensor))
			      (let ((stride (nth target-dim (tensor-stride tensor)))
				    (view (subscript-view (nth target-dim (tensor-view tensor)))))
				(lazy* stride (compute-stepby view))))))))

(defun expand-call-with-view-flatten
    (function
     tensors
     offset-place
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
			       collect (nth i s)))
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
				(nth k sizes)
				1)))
		     (list view))))))


(defmacro with-expand-init-tmp-form (offset-name-place tensors &body body)
  "Expands: initializing offsets, determining symbols form

Return: (values offsets-place form)"

  `(let ((,offset-name-place (tensor-gensym-list ,tensors)))
     ;; Initializing Offsets with 0
     `(let*-ignorable (,@(loop for name in ,offset-name-place
			       collect `(,name 0)))
	(locally (declare (type fixnum ,@,offset-name-place))
	  ,,@body))))


(defmacro with-shape-det-form (tensors &body body)
  `(let ((used-symbols))
     (mapc #'(lambda (tensor)
	       (mapc #'(lambda (s)
			 (when (symbolp s)
			   (push s used-symbols)))
		     (shape tensor)))
	   ,tensors)
     `(with-let-adjustable-symbols (,@used-symbols)
	,,@body)))


(defmacro with-expanding-explore-form ((tensors offset-places target-dim start-points end-points) &body body)
  ;; Set Strides At Runtime
  ;; Expand Loop
  `(let ((stride-places (tensor-gensym-list ,tensors))
	 (ith (gensym)))
     `(let (,@(loop for stride-place in stride-places ;; (place <- stride)
		     for tensor in ,tensors
		     collect `(,stride-place (nth ,,target-dim (list ,@(tensor-stride tensor))))))

	,@(expand-first-offset-adder
	   ,tensors
	   ,offset-places
	   stride-places
	   ,start-points)
	;; Expand Multi-Dimensional Looping Forms

	(loop for ,ith fixnum upfrom 0 below ,(car ,end-points)
	      ;; 1. Execute Operation
	      ;; 2. Adding Offsets
	      do (progn ,,@body)
	      unless (= ,ith (1- ,(car ,end-points)))
		;; Unless islast, expand it.
		do (progn ,@(expand-view-stride-adder ,offset-places stride-places ,target-dim ,tensors))))))

(defmacro with-expanding-explore-inlining ((tensors offset-places target-dim start-points end-points) &body body)
  `(with-expanding-explore-form* (,tensors ,offset-places ,target-dim ,start-points ,end-points) ,@body))

(defun call-with-view (function
		       tensors
		       &key
			 (at-least-dim 1)
		       &aux
			 (shape (shape (car tensors)))
			 (dims  (length shape)))
  "
## [function] call-with-view

(TODO)

"
  
  (declare ;;(optimize (speed 3))
	   (type function function)
	   (type list tensors shape)
	   (type fixnum at-least-dim dims))

  
  
  (assert (every #'(lambda (tensor) (shape-equal-list (butlast (shape tensor) at-least-dim) (butlast shape at-least-dim))) tensors)
	  nil
	  "call-with-view failed with assertion: All all tensors has the same dimensions of batch-area, butgot ~a."
	  (map 'list #'shape tensors)) ;; ... (1)

  (labels ((explore (rest-dim offsets-place &aux (target-dim (- dims rest-dim)))
	     (declare (type fixnum rest-dim target-dim)
		      (type list offsets-place))
	     ;; Exploring ND .. 3D 2D 1D

	     ;; When The Rest Form Can be flatten
	     (when (and (= at-least-dim 1) ;; Element-Wise Operation
			(apply #'order-reductable-p target-dim tensors) ;; check views
			(not (= rest-dim 0))) ;; If rest-dim = 0, use normal ver.
	       
	       (return-from explore
		 (expand-call-with-view-flatten
		  function
		  tensors
		  offsets-place
		  :dim-start-from target-dim)))
	     ;; Otherwise...

	     ;; Computing Multi-Dimensional Offsets
	     (let* ((start-points (loop for tensor in tensors
					collect
					(compute-visible-start-idx
					 (subscript-view (nth target-dim (tensor-view tensor)))
					 (nth target-dim (slot-value tensor 'orig-shape)))))
		    (end-points (loop for tensor in tensors
				      collect
				      (compute-visible-end-idx
				       (subscript-view (nth target-dim (tensor-view tensor)))
				       (nth target-dim (slot-value tensor 'orig-shape)))))
		    (axis-determined-p (every #'numberp end-points)))

	       (cond
		 ((<= rest-dim at-least-dim)
		  ;; funcall form
		  (expand-funcall-with-view
		   function
		   tensors
		   offsets-place
		   target-dim
		   rest-dim))
		 ((and axis-determined-p
		       (<= (the fixnum *unroll-threshold*) (the fixnum (car end-points))))

		  ;; Currently disabled
		  (with-expanding-explore-form
		      (tensors offsets-place target-dim start-points end-points)
		    (explore
		     (1- rest-dim)
		     offsets-place)))
		 (T
		  (with-expanding-explore-form
		      (tensors offsets-place target-dim start-points end-points)
		    (explore
		     (1- rest-dim)
		     offsets-place)))))))

    (with-shape-det-form tensors
      (with-expand-init-tmp-form offset-place tensors
	(explore dims offset-place)))))

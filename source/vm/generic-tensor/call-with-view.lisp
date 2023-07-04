

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
			 ;; non-reductable dim is: NOT(T) or NOT (:BROADCAST)
			 (or (not (eql (force-list v) t))
			     (not (eql (force-list v) :broadcast))
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
			      `(read-adjustable-symbol (nth ,target-dim-n (shape ,tensor)))
			      (let ((stride `(nth ,target-dim-n (list ,@(tensor-stride tensor))))
				    (view   `(subscript-view (nth ,target-dim-n (tensor-view ,tensor)))))
				(lazy* stride `(compute-stepby ,view))))))))

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
			       collect `(nth ,i (shape ,tensor))))
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
				`(read-adjustable-symbol ,(nth k sizes))
				`(compute-stepby
				  (subscript-view (nth ,target-dim (tensor-view ,tensor)))))))
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


(defmacro with-update-offset-place (offset-name-place tensors &body body &aux (tmp-space (gensym)))
  ""
  `(let ((,tmp-space ,offset-name-place)
	 (,offset-name-place (tensor-gensym-list ,tensors)))
     `(let*-ignorable (,@(loop for name in ,offset-name-place
			       for past-value in ,tmp-space
			       collect `(,name ,past-value)))
	(locally (declare (type fixnum ,@,offset-name-place))
	  ,,@body))))


(defmacro with-shape-det-form (tensors used-symbol-binding &body body)
  `(let ((used-symbols))
     (mapc #'(lambda (tensor)
	       (mapc #'(lambda (s)
			 (when (symbolp s)
			   (push s used-symbols)))
		     (shape tensor)))
	   ,tensors)
     `(progn
	(let ((,',used-symbol-binding ',used-symbols))
	  (declare (ignorable ,',used-symbol-binding))
	  ,,@body))))

(defmacro with-expanding-explore-form ((tensors offset-places target-dim start-points end-points) &body body &aux (endpoint-place (gensym)))
  ;; Set Strides At Runtime
  ;; Expand Loop
  `(let ((stride-places (tensor-gensym-list ,tensors))
	 (ith (gensym)))
     `(let* (,@(loop for stride-place in stride-places ;; (place <- stride)
		    for tensor in ,tensors
		    collect `(,stride-place (nth ,,target-dim (list ,@(tensor-stride tensor)))))
	    (,',endpoint-place ,(car ,end-points))
	    (,',endpoint-place (if (symbolp ,',endpoint-place)
				   (read-adjustable-symbol ,',endpoint-place)
				   ,',endpoint-place)))

	,@(expand-first-offset-adder
	   ,tensors
	   ,offset-places
	   stride-places
	   ,start-points)
	
	;; Expand Multi-Dimensional Looping Forms

	(loop for ,ith fixnum upfrom 0 below ,',endpoint-place
	      ;; 1. Execute Operation
	      ;; 2. Adding Offsets
	      do (progn ,,@body)
	      unless (= ,ith (1- ,',endpoint-place))
		;; Unless islast, expand it.
		do (progn ,@(expand-view-stride-adder ,offset-places stride-places ,target-dim ,tensors))))))

(defun update-calling-route (value)
  (push value cl-waffe2/vm.nodes::*call-with-view-route*))


(defmacro with-bind-shape (&body body)
  `(flet ((original-shape (tensor)
	    (translate-adjustable-shape (original-shape tensor)))
	  (shape (tensor)
	    (translate-adjustable-shape (shape tensor))))
     ,@body))


(defun call-with-view (function
		       tensors
		       &key
			 (at-least-dim 1)
		       &aux
			 (shape (shape (car tensors)))
			 (dims  (length shape)))
  "
## [function] call-with-view

```lisp
(call-with-view function tensors &key (at-least-dim 1))
```

The function `call-with-view` is a utility to expand view-considered `loop` iteration in the `:forward` expansion of `define-impl`.

(TODO: Example/Documents)

`function` [lambda] an lambda function which receives `variable1.view variable2.view ...` as arguments, returning an list being compiled.

`tensors` [list of abstracttensor] tensors to be called with.
`at-least-dim` [fixnum] ... kernel-size

See also:

`size-of`
`stride-of`
`offset-of`
"
  
  (declare ;;(optimize (speed 3))
	   (type function function)
	   (type list tensors shape)
	   (type fixnum at-least-dim dims))

  
  
  (assert (every #'(lambda (tensor) (shape-equal-list (butlast (shape tensor) at-least-dim) (butlast shape at-least-dim))) tensors)
	  nil
	  "call-with-view failed with assertion: All all tensors has the same dimensions of batch-area, butgot ~a."
	  (map 'list #'shape tensors)) ;; ... (1)

  (labels ((explore (rest-dim offsets-place used-symbols &aux (target-dim (- dims rest-dim)))
	     (declare (type fixnum rest-dim target-dim)
		      (type list offsets-place))
	     ;; Exploring ND .. 3D 2D 1D

	     ;; When The Rest Form Can be flatten
	     (when (and (= at-least-dim 1) ;; Element-Wise Operation
			(apply #'order-reductable-p target-dim tensors) ;; check views
			(not (= rest-dim 0))) ;; If rest-dim = 0, use normal ver.

	       (update-calling-route nil)
	       
	       (return-from explore
		 (expand-call-with-view-flatten
		  function
		  tensors
		  offsets-place
		  target-dim
		  :dim-start-from target-dim)))
	     
	     (update-calling-route rest-dim)
	     ;; Otherwise...

	     ;; Computing Multi-Dimensional Offsets
	     (let* ((start-points (loop for tensor in tensors
					collect
					`(compute-visible-start-idx
					  (subscript-view (nth ,target-dim (tensor-view ,tensor))))))
		    (end-points (loop for tensor in tensors
				      collect
				      `(compute-visible-end-idx
					(subscript-view (nth ,target-dim (tensor-view ,tensor)))
					(nth ,target-dim (original-shape ,tensor))))))
	       (cond
		 ((<= rest-dim at-least-dim)
		  ;; funcall form
		  (with-update-offset-place offsets-place tensors
		    (let ((stride-places (tensor-gensym-list tensors)))
		      `(let (,@(loop for stride-place in stride-places
				     for tensor in tensors
				     collect `(,stride-place (nth ,target-dim (list ,@(tensor-stride tensor))))))
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
			(tensors offsets-place target-dim start-points end-points)
		      (explore
		       (1- rest-dim)
		       offsets-place
		       used-symbols))))))))

    (with-shape-det-form tensors used-symbols
      (with-expand-init-tmp-form offset-place tensors
	(explore dims offset-place used-symbols)))))

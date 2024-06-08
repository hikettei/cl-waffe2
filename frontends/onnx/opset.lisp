
(in-package :cl-waffe2.frontends/onnx)

(defun broadcast-auto (a b)
  (if (= (wf/t:dims a) (wf/t:dims b))
      (wf:!view (wf:->mat a) (wf:broadcast-to b))
      a))

(macrolet ((def-elwise (name version op)
	     `(defop (,name ,version)
		  ((gph inputs attrs)
		    (declare (ignore attrs))
		    (assert (= 2 (length inputs)) () "Assertion Failed: ~a expects binary ops but got ~a" ,name inputs)
		    ;; [FIXME] Broadcasting semantic compapitibility?
		    ;; https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
		    (if (equal `() (wf/t:shape (first inputs)))
			(setf (first inputs) (make-tensor `(1) :initial-element (aref (wf/t:tensor-vec (first inputs)) 0))))
		    (if (equal `() (wf/t:shape (second inputs)))
			(setf (second inputs) (make-tensor `(1) :initial-element (aref (wf/t:tensor-vec (second inputs)) 0))))
		    (if (and (wf/t:scalar-p (car inputs)) (wf/t:scalar-p (second inputs)))
			(,op (car inputs) (second inputs))
			(let* ((ndim (max (wf/t:dims (car inputs)) (wf/t:dims (second inputs))))
			       (a (if (cl-waffe2/vm.generic-tensor:scalar-p (first inputs))
				      (first inputs)
				      (wf:!rankup (first inputs) (- ndim (wf/t:dims (car inputs))))))
			       (b (if (cl-waffe2/vm.generic-tensor:scalar-p (second inputs))
				      (second inputs)
				      (wf:!rankup (second inputs) (- ndim (wf/t:dims (second inputs))))))
			       (a (if (or (equal `(1) (wf/t:shape (first inputs))) (wf/t:scalar-p (first inputs)))
				      (wf:->scal a)
				      (broadcast-auto a b)))
			       (b (if (or (equal `(1) (wf/t:shape (second inputs))) (wf/t:scalar-p (second inputs)))
				      (wf:->scal b)
				      (broadcast-auto b a))))
			  (,op a b)))))))
  (def-elwise "Add" 1 wf:!add)
  (def-elwise "Sub" 1 wf:!sub)
  (def-elwise "Mul" 1 wf:!mul)
  (def-elwise "Div" 1 wf:!div))

(defop ("Conv" 1)
    ((cls inputs attrs)
      (let* ((data   (first inputs))
	     (kernel (second inputs))
	     (ndim   (length (cl-waffe2/vm.generic-tensor:shape data))))

	(when (null (gethash "kernel_shapes" attrs))
	  (setf (gethash "kernel_shapes" attrs) (cddr (cl-waffe2/vm.generic-tensor:shape kernel))))
	
	(when (gethash "auto_pad" attrs) ;; [WIP] we've not tested for the string attributes.
	  (cond
	    ((or (string= "SAME_UPPER" (gethash "auto_pad" attrs))
		 (string= "SAME_LOWER" (gethash "auto_pad" attrs)))
	     (error "NOT READY"))
	    ((string= "VALID" (gethash "auto_pad" attrs))
	     (setf (gethash "pads" attrs) (range 0 (- ndim 2))))
	    ((string= "NOTSET" (gethash "auto_pad" attrs)))
	    (T
	     (error "Value ~a in attribute \"auto_pad\" of operator Conv is invaild." (gethash "auto_pad" attrs)))))
	
	(setf (gethash "channels" attrs) (car (gethash "kernel_shapes" attrs)))

	(when (listp (gethash "pads" attrs))
	  (assert (= (mod (length (gethash "pads" attrs)) 2) 0))
	  
	  (let* ((pads (loop with size = (length (gethash "pads" attrs))
			     with mid = (/ size 2)
			     for i upfrom 0 below size by 2
			     for j = (+ mid i)
			     collect (list (nth i (gethash "pads" attrs)) (nth j (gethash "pads" attrs))))))
	    (setf data (wf:padding data `(t t ,@pads)))))

	(let ((model (wf/nn:Conv2D
		      (nth 1 (wf/t:shape data))
		      (nth 0 (wf/t:shape kernel))
		      (gethash "kernel_shapes" attrs)
		      :stride   (gethash "strides" attrs 1)
		      :padding  0
		      :dilation (gethash "dilations" attrs 1)
		      :groups   (gethash "group" attrs 1)
		      :bias     (= (length inputs) 3))))
	  (setf (wf/nn:weight-of model) kernel)
	  (when (= (length inputs) 3)
	    (setf (wf/nn:bias-of model) (third inputs)))
	  (wf/nodes:call model data)))))

(macrolet ((def-unary (name version op)
	     `(defop (,name ,version)
		  ((gph inputs attrs)
		    (declare (ignore attrs))
		    (,op (car inputs))))))

  (def-unary "Exp" 1 wf:!exp)
  (def-unary "Sqrt" 1 wf:!sqrt)
  (def-unary "Relu" 1 wf/nn:!relu)
  (def-unary "Sigmoid" 1 wf/nn:!sigmoid))

(defop ("LeakyRelu" 6)
    ((cls inputs attrs)
      (let ((alpha (gethash "alpha" attrs)))
	(wf/nn:!leaky-relu (car inputs) :negative-slope alpha))))

(defop ("Gemm" 1)
    ((cls inputs attrs)
      (assert (or (= (length inputs) 2) (= (length inputs) 3))
	      ()
	      "Assertion failed. Gemm should take two or three inputs but got: ~a" inputs)

      (let ((alpha (gethash "alpha" attrs))
	    (beta  (gethash "beta" attrs))
	    (transA (gethash "transA" attrs 0))
	    (transB (gethash "trabsB" attrs 0)))

	(let* ((a (if alpha (wf:!mul (car inputs) alpha) (car inputs)))
	       (b (second inputs))
	       ;; A/B has an inifinite-rank
	       (a (wf:!flexible a))
	       (b (wf:!flexible b))
	       (a (if (= 1 transA) (wf:!t a) a))
	       (b (if (= 1 transB) (wf:!t b) b))
	       (out (wf:!matmul a b))
	       (out (if (= (length inputs) 3)
			(wf:!add out (wf:!flexible (wf:!mul beta (third inputs))))
			out)))
	  out))))

(defop ("Shape" 1)
    ((cls inputs attrs)
      (declare (ignore attrs))
      (loop for s in (wf/t:shape (car inputs))
	    collect
	    (make-tensor s :dtype :int32))))

(defop ("Shape" 15)
    ((cls inputs attrs)
      (multiple-value-bind (start end)
	  (values (gethash "start" attrs) (gethash "end" attrs))
	(subseq (wf/t:shape (car inputs)) start end))))

(defop ("Unsqueeze" 1)
    ((cls inputs attrs)3
      (let ((x (wf:->mat (car inputs))))
	(dolist (axis (gethash "axes" attrs))
	  ;;(print "AXIS")
	  ;;(print x)
	  ;;(print axis)
	  (setf x (wf:!rankup x 1 :at (min (wf/t:dims x) axis))))
	x)))

(defop ("Concat" 1)
    ((cls inputs attrs)
      (setf inputs (alexandria:flatten inputs))
      (car (multiple-value-list (apply #'wf:!concatenate (gethash "axis" attrs) (map 'list (alexandria:compose #'wf:!flexible #'wf:->mat) inputs))))))
       
(defop ("Gather" 1)
    ((cls inputs attrs)
      (let* ((axis    (gethash "axis" attrs 0))
	     (data    (nth 0 inputs))
	     (indices (nth 1 inputs)))
	;; WIP
	(warn "Gather implementation may not be complete")
	;;(print axis)
	;;(print indices)
	;;(print data)
	(if (or (listp data) (wf/t::vec data))
	    (let ((i (wf/t:tensor-vec indices)))
	      (wf/t:make-tensor (nth (round (if (typep i 'real) i (aref i 0))) data)))
	    (wf/t:make-tensor (wf/vm:make-lazyaxis `(vref ,data (round (wf/t:tensor-vec ,indices)))))))))

(defop ("Cast" 1)
    ((cls inputs attrs)
      (warn "Cast implementation may not be complete")
      (car inputs)))

(defop ("Range" 1)
    ((cls inputs attrs)
      (declare (ignore attrs))
      (let ((out (make-input `((- (wf/t:tensor-vec ,(nth 1 inputs)) (wf/t:tensor-vec ,(nth 0 inputs)))) nil)))
	(wf:lazy-index-components
	 #'(lambda (i) (* i (wf/t:tensor-vec (nth 2 inputs))))
	 out))))

(defop ("Reduce" 1)
    ((gph inputs attrs &key (reduce #'wf:!sum))
      (let ((axis (gethash "axes" attrs 0)))
	(car
	 (multiple-value-list
	  (funcall
	   reduce
	   (car inputs)
	   :axis axis
	   :keepdims
	   (if (= (gethash "keepdims" attrs 1) 1)
	       t
	       nil)))))))

(defun topk (k f)
  #'(lambda (&rest args)
      (let ((topN (sort args f))
	    (K (wf/vm:maybe-observe-axis k)))
	(apply #'values (loop for n upfrom 0 below K collect (nth n topN))))))

(defun topk-indices (k f)
  #'(lambda (&rest args)
      (let ((topN (sort args f))
	    (K (wf/vm:maybe-observe-axis k)))
	(let ((pos (loop for n upfrom 0 below K collect (nth n topN))))
	  (apply #'values (map 'list #'(lambda (x) (position x args :test #'=)) pos))))))

(defun !topk (tensor k axis f)
  (wf:lazy-reduce (topk K f) tensor :reduce-to k :axis axis))

(defun !topk-indices (tensor k axis f)
  (wf:lazy-reduce (topk-indices K f) tensor :reduce-to k :axis axis))

(defop ("TopK" 1)
    ((cls inputs attr)
      (let* ((largest (gethash "largest" attr 1))
	     (axis    (gethash "axis" attr -1))
	     (sorted  (gethash "sorted" attr 1))
	     (f       (if (= largest 1)
			  #'>
			  #'<)))
	(setf axis (1- (wf/t:dims (car inputs))))
	(let ((out (!topk (car inputs) (second inputs) axis f))
	      (indices (!topk-indices (car inputs) (second inputs) axis f)))
	  (values out indices)))))


(macrolet ((defreduce (name op)
	     `(defop (,name 0)
		  ((gph inputs attrs)
		    (let ((r (get-converter "Reduce" (gp-opset-version gph))))
		      (funcall r gph inputs attrs :reduce ,op))))))
  (defreduce "ReduceMax"  #'wf:!max)
  (defreduce "ReduceMin"  #'wf:!min)
  (defreduce "ReduceSum"  #'wf:!sum)
  (defreduce "ReduceMean" #'wf:!mean)
  ;; ReduceProd
  ;; ReduceLogSumExp
  )

(defop ("Less" 9)
    ((cls inputs attrs)
      (declare (ignore attrs))
      (if (and (wf/t:scalar-p (car inputs)) (wf/t:scalar-p (second inputs)))
	  (wf:->scal (wf:A<B (wf:->mat (first inputs)) (wf:->mat (second inputs))))
	  (if (wf/t:scalar-p (second inputs))
	      (wf:A<scal (car inputs) (second inputs))
	      (if (wf/t:scalar-p (car inputs))
		  (wf:A>=scal (second inputs) (wf/t:tensor-vec (car inputs)))
		  (wf:A<B (car inputs) (second inputs)))))))

(defop ("Equal" 9)
    ((cls inputs attrs)
      (declare (ignore attrs))
      (if (wf/t:scalar-p (second inputs))
	  (wf:A=B (car inputs) (wf:!flexible (wf:->mat (second inputs))))
	  (if (wf/t:scalar-p (car inputs))
	      (wf:A=B (second inputs) (wf:!flexible (wf:->mat (car inputs))))
	      (wf:A=B (car inputs) (second inputs))))))

(defop ("Where" 9)
    ((cls inputs attrs)
      ;; condition: True=1, False=0
      (let* ((true-mask (car inputs))
	     (false-mask (wf:!mul -1 (wf:!sub (car inputs) 1))) ;; bitmask
	     (true-values (wf:!mul (nth 1 inputs) (wf:!flexible (wf:!flatten true-mask))))
	     (false-values (wf:!mul (nth 2 inputs) (wf:!flexible (wf:!flatten false-mask)))))
	(wf:!add (wf:!flexible true-values) (wf:!flexible false-values)))))


(defop ("Pow" 13)
    ((cls inputs attrs)
      (declare (ignore attrs))
      (wf:!expt (car inputs) (second inputs))))

(defop ("Constant" 9)
    ((cls inputs attrs)
      (let ((val (gethash "value" attrs)))
	(when val
	  (if (numberp val)
	      (make-tensor val)
	      (if (and (not (stringp val)) (arrayp val))
		  (change-facet val :direction 'AbstractTensor)
		  (progn
		    (warn "cl-waffe2 attempts somehow convert ~a to AbstractTensor in ConstantNode using change-facet method, but this may failed."
			  val)
		    (change-facet val :direction 'AbstractTensor))))))))

(defop ("ConstantOfShape" 9)
    ((cls inputs attrs)
      (block constantofshape
	(let* ((value (or (and (gethash "value" attrs) (change-facet (gethash "value" attrs) :direction 'AbstractTensor)) (make-tensor 0.0)))
	       (input (if (typep (car inputs) 'wf/t::AbstractTensor)
		          (car inputs)
			  (if (listp (car inputs))
			      (return-from ConstantOfShape (car inputs))
			      (change-facet (car inputs) :direction 'AbstractTensor)))))
	  (wf:!add (wf:!mul 0 input) (wf:!flexible value))))))

(defop ("Slice" 10)
    ((cls inputs attrs)
      (declare (ignore attrs))
      (multiple-value-bind (start end axes steps)
	  (values (nth 1 inputs) (nth 2 inputs) (nth 3 inputs) (nth 4 inputs))
	(assert (wf/t::vec axes) () "[WIP] Slice: Dynamic axes is not implemented yet. got ~a" axes)
	(let* ((data (nth 0 inputs))
	       (normalized-axes
		 (loop for axis across (wf/t::vec axes)
		       if (>= axis 0)
			 collect axis
		       else
			 collect (+ 1 (wf/t:dims data) axis)))
	       (normalized-views
		 (loop for nth upfrom 0 below (wf/t:dims data)
		       if (find nth normalized-axes :test #'=)
			 collect `(,(wf/vm:make-lazyaxis `(wf/t:vref ,start ,nth))
				   ,(wf/vm:make-lazyaxis `(wf/t:vref ,end ,nth))
				   ,(wf/vm:make-lazyaxis `(wf/t:vref ,steps ,nth)))
		       else
			 collect t)))
	  (car (multiple-value-list (apply #'wf:!view data normalized-views)))))))	      

(defop ("Transpose" 1)
    ((cls inputs attrs)
      (let ((perm (gethash "perm" attrs)))
	(assert (listp perm) () "[WIP] Transpose: perm must be static and given as a list. but got ~a" perm)
	(wf:!permute (car inputs) (apply #'wf:torch-order perm)))))

(defop ("Expand" 8)
    ((cls inputs attrs)
      ;;(warn "Expand is not complete. ~a" (second inputs))
      (let* ((out (wf:!rankup (wf:->mat (car inputs)) (- (wf/t:dims (car inputs)) (length (second inputs)))))
	     (s   (loop for o in (wf/t:shape out)
			for e in (second inputs)
			if (= 1 (wf/t:tensor-vec e))
			  collect t
			else
			  collect `(:broadcast ,(wf/t:tensor-vec e)))))
	(car (multiple-value-list (apply #'wf:!view out s))))))

(defop ("Tile" 6)
    ((cls inputs attrs)
      (wf:!tile (car inputs) (second inputs))))

(defop ("Reshape" 1)
    ((cls inputs attrs)
      (apply #'wf:!reshape (car inputs) (gethash "shape" attrs))))

(defop ("Reshape" 5)
    ((cls inputs attrs)
      (apply
       #'wf:!reshape
       (car inputs)
       (loop for nth upfrom 0 below (length (wf/t:shape (second inputs)))
	     collect
	     (wf/vm:make-lazyaxis `(wf/t:vref ,(second inputs) ,nth))))))

(defop ("Flatten" 1)
    ((cls inputs attrs)
      (let* ((axis (gethash "axis" attrs))
	     (reshaped (list
			(wf/vm:make-lazyaxis
			 `(*
			   ,@(loop for i upfrom 0 below axis
				   collect
				   (nth i (wf/t:shape (car inputs))))))
			(wf/vm:make-lazyaxis
			 `(*
			   ,@(loop for i upfrom axis below (wf/t:dims (car inputs))
				   collect
				   (nth i (wf/t:shape (car inputs)))))))))
	(apply #'wf:!reshape (car inputs) reshaped))))

(defun resize-op-11-13-common (cls size inputs attrs)
  (let* ((roi (nth 1 inputs))
	 (roi (if (and roi (= (nth 0 (wf/t:shape roi)) 0))
		  nil
		  roi))
	 (ndims (wf/t:dims (car inputs)))
	 (mode (if (gethash "mode" attrs)
		   (map 'string #'code-char (gethash "mode" attrs))
		   "nearest"))
	 (method
	   (macrolet ((of (value) `(string= mode ,value)))
	     (cond
	       ((of "nearest") "nearest_neighbor")
	       ((of "linear") "linear")
	       ((of "cubic") "cubic")
	       (T (error "~a is not valid." mode)))))
	 (coord-trans
	   (if (gethash "coordinate_transformation_mode" attrs)
	       (map 'string #'code-char (gethash "coordinate_transformation_mode" attrs))
	       "half_pixel"))
	 (nearest-mode
	   (if (gethash "nearest_mode" attrs)
	       (map 'string #'code-char (gethash "nearest_mode" attrs))
	       "round_prefer_floor"))
	 (alpha (gethash "cubic_coeff_a" attrs -0.75))
	 (exclude (gethash "exclude_outside" attrs 0))
	 (extrapolation-value (gethash "extrapolation_value" attrs 0.0))
	 (roi (when roi (error "resize-op-11-13-common: roi=True is not implemented yet.")))
	 (out-size (if size size (error "size=nil is not impllemented yet."))))
				  

    (labels
	((nearest-gather (X x-out y-out)
	   (wf:!view (wf:!view x t t y-out t) t t t x-out))
	 (nearest-mode (x-resized x-len)
	   (macrolet ((of (value) `(string= nearest-mode ,value)))
	     (cond
	       ((of "round_prefer_ceil") (wf:!- x-len 0.5) (error "round_prefer_floor"))
	       ((of "round_prefer_floor") (wf:lazy #'floor (wf:!- x-len 0.5)))
	       ((of "floor") (wf/vm:make-lazyaxis `(floor ,x-len)))
	       ((of "ceil") (error "ceil"))
	       (T (error "unknown nearest-mode ~a" nearest-mode)))))
	 (coordinate-trans (x-out y-out out-shape scales)
	   (macrolet ((of (value) `(string= coord-trans ,value)))
	     (multiple-value-bind (x-out y-out)
		 (values
		  (cond
		    ((of "asymmetric")
		     (values
		      (wf:!/ x-out (wf:!- (wf:!view scales 0)))
		      (wf:!/ y-out (wf:!- (wf:!view scales 1)))))
		    (T
		     (error "coordinate-trans: ~a is not implemented" coord-trans))))
	       (values x-out y-out)))))

      (let ((x-out (wf:lazy-index-components #'(lambda (i) i) (make-input `(,(wf/vm:make-lazyaxis `(wf/t:vref ,out-size (+ ndim -1)))) nil)))
	    (y-out (wf:lazy-index-components #'(lambda (i) i) (make-input `(,(wf/vm:make-lazyaxis `(wf/t:vref ,out-size (+ ndim -2)))) nil))))
	(case ndims
	  (4
	   (macrolet ((of (value) `(string= mode ,value)))
	     (cond
	       ((of "nearest")
		(multiple-value-bind (x-out y-out)
		    (coordinate-trans x-out y-out out-size (nth 2 inputs))
		  (nearest-gather
		   (car inputs)
		   (nearest-mode x-out (car (last (wf/t:shape (car inputs)))))
		   (nearest-mode y-out (car (last (wf/t:shape (car inputs))))))))
	       (T (error "~a is not implemented" mode)))))
	  (T (error "resize-op-11-13-common: ndims = ~a is not implemented" ndims)))))))

(defop ("Resize" 11)
    ((cls inputs attrs)
      (let* ((scale (nth 2 inputs))
	     (size (if (= (length inputs) 4)
		       (nth 3 inputs)
		       (wf:!mul (nth 0 inputs) (wf:!flexible scale)))))
	(car (multiple-value-list (resize-op-11-13-common cls size inputs attrs))))))

      
(defop ("NonMaxSuppression" 11)
    ((cls inputs attr)
      (wf:!add (wf/t:make-input `(10 3) nil) (wf:->scal (wf:!view (car inputs) 0)))
      ))


(in-package :cl-waffe2/vm.generic-tensor)

;; **THE CODES BELOW MUST BE OPTIMIZED**
(defstruct (ViewInstruction
	    (:constructor
		make-viewinstruction (offset size)))
  (size 0 :type fixnum)
  (offset 0 :type fixnum))

;; TODO: With Undetermined symbols
;; TOOD: Create a view with undetermined symbols
(defun call-with-view (function
		       &rest tensors
		       &aux
			 (result)
			 (shape (shape (car tensors)))
			 (dims  (length (shape (car tensors)))))
  "Unrolling...
What kind of tensors could be called together?
-> Tensors whose dimensions and shapes are the same.

(sin x)
(adds x y) <- Broadcast it.
(gemm x y z)"

  ;; Detect call-with-view-ext

  (assert (every #'(lambda (tensor) (equal (shape tensor) shape)) tensors)
	  nil
	  "Assertion Failed because the number of shapes: ~a doesn't match."
	  (map 'list #'shape tensors))

  ;; TODO: Orders match?

  ;; Unroll Untill 2D/1D

  (labels ((explore (rest-dim offsets &aux (processing-dim (- dims rest-dim)))
	     (cond
	       ((= rest-dim 1)
		;; 1D

		;; koreha kasu (tmp)
		(let ((args (loop for tensor in tensors
				  for k upfrom 0
				  collect (make-viewinstruction
					   (+ (nth k offsets)
					      (view-startindex (nth processing-dim (tensor-view tensor)) 0))
					   (- (view-endindex (nth processing-dim (tensor-view tensor))
							     (nth processing-dim (slot-value tensor 'orig-shape)))
					      (view-startindex (nth processing-dim (tensor-view tensor))
							       0))))))

		  ;; (if shape is determined...
		  ;; TODO: (if shape is undetermined -> loop it.
		  (push (apply function args) result)))
	       (T
		;; 3D, 4D, ...
		(let ((new-offsets (copy-list offsets))
		      (start-points (loop for tensor in tensors
					  collect (view-startindex
						   (nth processing-dim (slot-value tensor 'view)) 0)))
		      (loop-iternum (view-endindex (nth processing-dim (slot-value (car tensors) 'view))
						   (nth processing-dim (slot-value (car tensors) 'orig-shape)))))

		  ;; Adds First-Offset
		  (loop for tensor in tensors
			for offset in start-points
			for past-offset in new-offsets
			for k fixnum upfrom 0
			do (setf (nth k new-offsets)
				 (+ past-offset
				    (* (nth processing-dim (tensor-stride tensor))
				       offset))))

		  (dotimes (i loop-iternum)
		    (explore (1- rest-dim) new-offsets)

		    (loop for tensor in tensors
			  for k fixnum upfrom 0
			  do (incf (nth k new-offsets) (nth processing-dim (tensor-stride tensor))))))))))
    
    (explore
     dims
     (make-list (length tensors) :initial-element 0))
    
    `(progn ,@(reverse result))))

;; View 書き直す

(deftype subscript-t ()
  "An list of types which is allowed to become subscripts of the function view."
  `(or fixnum list null t))

(deftype subscript-syntax ()
  `(member :index :t :slice :slice-step :indices :tflist :broadcast :repeat))


(defstruct (subscript
	    (:print-function
	     (lambda (subscript stream depth)
	       (declare (ignore depth))
	       (format stream "<~a~a"
		       (subscript-view subscript)
		       (if (typep (subscript-char subscript) 'symbol)
			   (format nil ">~a∊~a"
				   (subscript-char subscript)
				   (subscript-constraints subscript))
			   ">")))))
  (constraints nil :type list) ;; x is in (10 30) t ... inf, min=0
  (view nil :type subscript-t)
  (char nil)
  (determined-p nil :type boolean))

(defmacro with-viewcase ((view)
			 &key
			   index
			   tcase
			   slice
			   keyword)
  `(typecase ,view
     (fixnum ,index)
     (list
      (cond
	((typep (car ,view) 'fixnum)
	 ,slice)
	((typep (car ,view) 'keyword)
	 ,keyword)
	(T (error "Unknown view ~a" ,view))))
     (T ,tcase)))

;; TODO: from view to view
(declaim (ftype (function (subscript-t) subscript-syntax) viewtype))
(defun viewtype (view)
  (with-viewcase (view)
		 :index :index
		 :tcase :t
		 :slice (cond
			  ((= (length view) 2) :slice)
			  ((= (length view) 3) :slice-step)
			  (T (error "(start stop) or (start stop step)")))
		 :keyword (case (car view)
			    (:indices :indices)
			    (:broadcast :broadcast)
			    (:repeat :repeat)
			    (T (error ":indices :broadcast :repeat")))))

;; 一般化したい
(defun compute-visible-start-idx (view size)
  (declare (ignore size))
  (case (viewtype view)
    (:index 0)
    (:t     0)
    (:slice (car view))
    (:slice-step (car view))
    (:indices 0)
    (:tflist  0)
    (:broadcast 0)
    (:repeat 0)))

;; :Lazy-Eval it
(defun compute-visible-end-idx (view size)
  (case (viewtype view)
    (:index 1)
    (:t     size)
    (:slice (second view))
    (:slice-step (second view))
    (:indices (length (cdr view)))
    (:tflist  size)
    (:broadcast (second view))
    (:repeat `(* ,(second view) size))))


(defun compute-visible-shape (orig-shape view)
  (loop for o in orig-shape
	for v in view
	collect (let ((end   (compute-visible-end-idx   v o))
		      (start (compute-visible-start-idx v 0)))
		  (cond
		    ((and (typep start 'fixnum)
			  (= start 0))
		     end)
		    ((and (typep end 'fixnum)
			  (typep start 'fixnum))
		     `(- ,end ,start))
		    (T
		     `(- ,end ,start))))))


(defun parse-absolute (index size)
  (typecase size
    (fixnum (if (< index 0)
		(+ index size)
		index))
    (symbol
     (if (< index 0)
	 `(+ ,index ,size)
	 index))))

(defgeneric step-subscript (before-type after-type before after size))

(defmethod step-subscript ((x (eql :t))
			   (y (eql :index))
			   before
			   after
			   size)
  "Tensor[t][Index] but the size is undetermined. (e.g.: Tensor = (a b))
Return: (values after-view error)"
  (let ((index (parse-absolute (subscript-view after) size)))
    (setf (subscript-char after) size
          (subscript-constraints after) `(,index t)
	  (subscript-view after) index)
    after))

(defmethod step-subscript ((x (eql :t))
			   (y (eql :t))
			   before
			   after
			   size)
  "Tensor[t][t]"
  (setf (subscript-char after) size
        (subscript-constraints after) `(0 t))
  after)
	      
(defmethod step-subscript ((x (eql :t))
			   (y (eql :slice))
			   before
			   after
			   size)
			   
  "Tensor[t][2:4]"
  (let* ((view (subscript-view after))
	 (view (map 'list #'(lambda (v) (parse-absolute v size)) view)))
    (setf (subscript-char after) size
	  (subscript-constraints after) `(,(second view) t)
	  (subscript-view after) view)
    after))

	      
(defmethod step-subscript ((x (eql :t))
			   (y (eql :slice-step))
			   before
			   after
			   size)
			   
  "Tensor[t][2:4::-1]"
  (let* ((view (subscript-view after))
	 (view (map 'list #'(lambda (v) (parse-absolute v size)) (butlast view))))
    (setf (subscript-char after) size
	  (subscript-constraints after) `(,(second view) t)
	  (subscript-view after) `(,@view ,(third (subscript-view after))))
    after))

(defmethod step-subscript ((x (eql :t))
			   (y (eql :indices))
			   before
			   after
			   size)
			   
  "Tensor[t][:indices 1 2 3...]"
  (let* ((view (subscript-view after)))
    (if (typep view 'list)
	nil) ;; maximize view ...
    ;; TMP
    (setf (subscript-char after) size
	  (subscript-constraints after) `(0 t))
    after))

(defmethod step-subscript ((x (eql :t))
			   (y (eql :broadcast))
			   before
			   after
			   size)
			   
  "Tensor[t][:broadcast n]"
  (setf (subscript-char after) size
	(subscript-constraints after) `(1 1))
  after)

(defmethod step-subscript ((x (eql :t))
			   (y (eql :repeat))
			   before
			   after
			   size)
			   
  "Tensor[t][:repeat n]"
  (setf (subscript-char after) size
	(subscript-constraints after) `(0 t))
  after)

(defun preprocess-subscript (dim tensor size subscript)
  (declare (type fixnum dim)
	   (type AbstractTensor tensor))
  (let* ((determined-p (typep size 'fixnum))
	 (projected-p  (slot-value tensor 'projected-p))
	 (past-view (or (when projected-p
			  (nth dim (tensor-view tensor)))
			(make-subscript
			 :determined-p t
			 :view t)))
	 (view (make-subscript
		:determined-p determined-p
		:view subscript)))

    (step-subscript
     (viewtype (subscript-view past-view))
     (viewtype (subscript-view view))
     past-view
     view
     size)))

(defun parse-view-subscripts (tensor subscripts)
  "
TensorのShape...`((10 10) (10 10) (10 a)) ...
未決定のシンボルがあったら、それに対する制約を求める。
shape = (batch-size), viewが`(0 10)なら, batch-size>=10.
STEP=-1など

subscript = fixnum, t ,list
list = (0 10)
       or (0 10 -1)
       or (:tflist 0 10 0 10)
       or (:indices 0 1 2 3 4 5)
       or (:broadcast 10)
       or (:repeat 10)
"
  (loop with shape = (slot-value tensor 'orig-shape)
        for i fixnum upfrom 0 below (length shape)
	;; multiple-value-bind (res errror)
	collect (preprocess-subscript i
				      tensor
				      (nth i shape)
				      (or (nth i subscripts) t))))


(defun call-with-view (function &rest tensors)
  
  )

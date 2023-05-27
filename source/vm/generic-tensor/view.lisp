
(in-package :cl-waffe2/vm.generic-tensor)

;; **THE CODES BELOW MUST BE OPTIMIZED**
(defstruct (ViewInstruction
	    (:constructor
		make-viewinstruction (offset size)))
  (size)
  (offset))

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
	collect (let ((end   (compute-visible-end-idx (subscript-view v) o))
		      (start (compute-visible-start-idx (subscript-view v) 0)))
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
  "Tensor[t][Index]
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

(defun preprocess-subscript (dim tensor size past-view subscript)
  (declare (type fixnum dim)
	   (type AbstractTensor tensor)
	   (ignore dim))
  (let* ((determined-p (typep size 'fixnum))
	 (projected-p  (slot-value tensor 'projected-p))
	 (past-view (or (when projected-p past-view)
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

(defun parse-view-subscripts (tensor past-view subscripts)
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
				      (nth i past-view)
				      (or (nth i subscripts) t))))

(defmacro %* (x y)
  `(the fixnum (* (the fixnum ,x) (the fixnum ,y))))

(defmacro %+ (x y)
  `(the fixnum (+ (the fixnum ,x) (the fixnum ,y))))

(defun call-with-view  (function
		        tensors
			&key
			  (at-least-dim 1)
			&aux
			  (shape (shape (car tensors)))
			  (dims  (length shape)))
  ;; Continue from here.


  (assert (every #'(lambda (tensor) (equal (shape tensor) shape)) tensors)
	  nil
	  "Assertion Failed because the number of shapes: ~a doesn't match."
	  (map 'list #'shape tensors)) ;; ... (1)

  
  (labels ((expand-with-function (target-dim offsets)
	     ;; currently only for 1d
	     (apply function
		    (loop for k fixnum upfrom 0
			  for tensor in tensors
			  collect (make-viewinstruction
				   `(nth ,k ,offsets)
				   (nth target-dim (shape tensor))))))
	   (explore (rest-dim offsets &aux (target-dim (- dims rest-dim)))
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
	       (declare (ignore axis-determined-p))
	       ;; When axis-determined-p is nil expand with loop for parts
	       ;; is t -> Unroll

	       (let ((stride-place (gensym)))
		 `(let ((,stride-place (list ,@(loop for tensor in tensors
						     collect (nth target-dim (tensor-stride tensor))))))
		    (declare (type list ,stride-place))
		    ;; adds offset
		    ,@(loop for k upfrom 0
			    for tensor in tensors
			    collect `(setf (nth ,k ,offsets) (%+ (%* (nth ,k ,stride-place)
								     ,(nth k start-points))
								 (nth ,k ,offsets))))
		    

		    ,@(if nil;;axis-determined-p ;; need unroll-p
			  (loop for nth fixnum upfrom 0 below (car end-points)
				collect (prog1
					    (if (<= rest-dim at-least-dim)
						(expand-with-function target-dim offsets)
						(explore
						 (1- rest-dim)
						 offsets))
					  (loop for k upfrom 0
						for tensor in tensors
						;; If we have a broadcasted axis here, freeze the axis (TODO)
						collect `(incf (the fixnum (nth ,k ,offsets)) (the fixnum (nth ,k ,stride-place))))))

			  (let ((nth (gensym)))
			    `((loop for ,nth fixnum upfrom 0 below ,(car end-points)
				    collect ,(prog1
						 (if (<= rest-dim at-least-dim)
						     (expand-with-function target-dim offsets)
						     (explore
						      (1- rest-dim)
						      offsets))
					       (loop for k upfrom 0
						     for tensor in tensors
						     ;; If we have a broadcasted axis here, freeze the axis (TODO)
						     collect `(incf (the fixnum (nth ,k ,offsets)) (the fixnum (nth ,k ,stride-place))))))))))))))

    (let ((offset-place (gensym))
	  (nondeterministic-symbols))
      (mapc #'(lambda (tensor)
		(loop for i upfrom 0
		      for shape in (shape tensor)
		      if (not (numberp shape))
			do (push `(,shape (nth ,i (shape ,tensor))) nondeterministic-symbols)))
	    tensors)
      `(let ((,offset-place (make-list ,(length tensors) :initial-element 0)))
	 (let (,@nondeterministic-symbols)
	   (declare (type fixnum ,@(map 'list #'car nondeterministic-symbols)))
	   ,(explore
	     dims
	     offset-place))))))



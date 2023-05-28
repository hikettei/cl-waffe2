
(in-package :cl-waffe2/vm.generic-tensor)

(defparameter *unroll-threshold* 10 "Unroll if iternum is below this threshold")

;; TO Add: ViewInstruction2D to implement matmul
(defstruct (ViewInstruction
	    (:constructor
		make-viewinstruction (offset size by)))
  (size) ;; contains list, fixnum...
  (by)
  (offset))

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
	(T (error "Unknown view: ~a" ,view))))
     (T
      (if (eql ,view t)
	  ,tcase
	  (error "Unknown view: ~a" ,view)))))

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

(defun compute-visible-actual-shape (tensor)
  (loop for v in (tensor-view tensor)
	for s in (slot-value tensor 'orig-shape)
	collect (- (compute-visible-end-idx-actual
		    (subscript-view v)
		    s)
		   (compute-visible-start-idx
		    (subscript-view v)
		    s))))

(defun compute-visible-start-idx (view size)
  (declare (ignore size))
  (case (viewtype view)
    (:index view)
    (:t     0)
    (:slice (car view))
    (:slice-step (car view))
    (:indices 0)
    (:tflist  0)
    (:broadcast 0)
    (:repeat 0)))

(defun compute-visible-end-idx (view size)
  (case (viewtype view)
    (:index (1+ view))
    (:t     size)
    (:slice (second view))
    ;; FIXME: Should be divided :slice-step
    (:slice-step (round (/ (second view) (abs (third view)))))
    (:indices (length (cdr view)))
    (:tflist  size)
    (:broadcast (second view))
    (:repeat `(* ,(second view) size))))

(defun compute-visible-end-idx-actual (view size)
  (case (viewtype view)
    (:index (1+ view))
    (:t     size)
    (:slice (second view))
    ;; FIXME: Should be divided :slice-step
    (:slice-step (round (/ (second view) (abs (third view)))))
    (:indices (length (cdr view)))
    (:tflist  size)
    (:broadcast 1)
    (:repeat (second view))))


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

;; [T] -> Any
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
        (subscript-constraints after) `(0 t)) ;; using before, adjust constraints
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

;; Tensor[Index] -> Any
;; Tensor = (10, 10), Tensor[0] = (1, 10)

(defmethod step-subscript ((x (eql :index))
			   (y (eql :t))
			   before
			   after
			   size)
  "Tensor[0][t] is the same as just doing Tensor[t][0]"
  (step-subscript :t :t before after size))


(defmethod step-subscript ((x (eql :index))
			   (y (eql :index))
			   before
			   after
			   size)
  "Tensor[0][1]"
  (step-subscript :t :index before after size))

(defmethod step-subscript ((x (eql :index))
			   (y (eql :slice))
			   before
			   after
			   size)
  "Tensor[0][0:10]"
  (step-subscript :t :slice before after size))

(defmethod step-subscript ((x (eql :index))
			   (y (eql :slice-step))
			   before
			   after
			   size)
  "Tensor[0][0:10::2]"
  (step-subscript :t :slice-step before after size))

(defmethod step-subscript ((x (eql :index))
			   (y (eql :indices))
			   before
			   after
			   size)
  "Tensor[1][:indices 0 1 2]"
  (step-subscript :t :indices before after size))

(defmethod step-subscript ((x (eql :index))
			   (y (eql :tflist))
			   before
			   after
			   size)
  "Tensor[1][:tflist t]"
  (step-subscript :t :tflist before after size))

(defmethod step-subscript ((x (eql :index))
			   (y (eql :broadcast))
			   before
			   after
			   size)
  "Tensor[1][:broadcast 10]"
  (step-subscript :t :broadcast before after size))

(defmethod step-subscript ((x (eql :index))
			   (y (eql :repeat))
			   before
			   after
			   size)
  "Tensor[1][:repeat 10]"
  ;; The same as just doing broadcast
  (step-subscript :t :broadcast before after size))


;; Tensor[0:10] -> Any

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :t))
			   before
			   after
			   size)
  "Tensor[0:10][t] is the equivalent to Tensor[t][0:10]"
  (step-subscript :t :slice before after size))

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :index))
			   before
			   after
			   size)
  "Tensor[2:10][0] Add offset to index.

I.e.: From viewpoint of x-orig, x-orig[1:5][0] is x-orig[1]"
  (incf (subscript-view after) (car (subscript-view before)))
  (step-subscript :t :slice before after size))

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :slice))
			   before
			   after
			   size)
  "Tensor[2:10][0:3]" ;; Just Adding Offsets
  (incf (car (subscript-view after))
	(car (subscript-view before)))
  (incf (second (subscript-view after))
	(car (subscript-view before)))
  (step-subscript :t :slice before after size))

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :slice-step))
			   before
			   after
			   size)
  "Tensor[2:10][0:3::2]"
  (step-subscript :slice :slice before after size))

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :indices))
			   before
			   after
			   size)
  "Tensor[2:10][:indices 10 20]"
  (error "TODO [2:10] -> [:indices 10 20]")
  ;; [:indices 10 20] + [2, 10][0]
  )

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :tflist))
			   before
			   after
			   size)
  "Tensor[2:10][:tflist t t nil]"
  ;; padding with NIL
  (error "TODO [2:10] -> [:tflist t nil...]"))

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :broadcast))
			   before
			   after
			   size)
  "Tensor[2:10][:broadcast 10]"
  (error "TODO [2:10] -> [:broadcast 10]"))

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :repeat))
			   before
			   after
			   size)
  "Tensor[2:10][:repeat 10]"
  (error "TODO [2:10] -> [:repeat 10]"))



;; Tensor[:broadcast n] -> Any
(defmethod step-subscript ((x (eql :broadcast))
			   (y (eql :t))
			   before
			   after
			   size)
  "Tensor[:broadcast 10][t]"
  (step-subscript :t :broadcast before after size))

(defmethod step-subscript ((x (eql :broadcast))
			   (y (eql :index))
			   before
			   after
			   size)
  "Tensor[:broadcast 10][1]"
  ;; TODO: ADD CONSTRAINTS
  (setf (subscript-view after) 0)
  (step-subscript :t :index before after size))


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

Return: List[SubScript]
"
  (loop with shape = (slot-value tensor 'orig-shape)
        for i fixnum upfrom 0 below (length shape)
	collect (preprocess-subscript i
				      tensor
				      (nth i shape)
				      (nth i past-view)
				      (or (nth i subscripts) t))))

(defmacro %* (x y)
  `(the fixnum (* (the fixnum ,x) (the fixnum ,y))))

(defmacro %+ (x y)
  `(the fixnum (+ (the fixnum ,x) (the fixnum ,y))))

(defun shape-equal (a b)
  "a=1, b=k => T
   a=1, b=2 => NIL
..."
  (if (and (numberp a)
	   (numberp b))
      (= (the fixnum a) (the fixnum b))
      (if (and (symbolp a)
	       (symbolp b))
	  (eql a b)
	  t)))

(defun shape-equal-list (list1 list2)
  (every #'shape-equal list1 list2))


;; Translate :tflist into :indices
(defun expand-view-stride-adder (nth offsets strides target-dim tensors)
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
		     `(incf (the fixnum (nth ,k ,offsets))
			    (the fixnum (nth ,k ,strides))))
		    ((eql viewtype :slice-step)
		     `(incf (the fixnum (nth ,k ,offsets))
			    (%* ,(third view)
				(the fixnum (nth ,k ,strides)))))
		    ((eql viewtype :repeat)
		     (error ":REPEAT IS NOT IMPLEMENTED"))
		    ((eql viewtype :indices)
		     (print nth)
		     (error ":INDICES IS NOT IMPLEMENTED"))
		    ((eql viewtype :tflist)
		     (error ":TFLIST IS NOT IMPLEMENTED"))
		    (T
		     (error "Unknown keyword ~a" viewtype))))))

(defun compute-stepby (view)
  ;; view ... list
  (case (viewtype view)
    (:slice-step (third view))
    (:broadcast 0)
    (T 1)))
    
;; :indices, :tflist -> wrap by call-with-view-ext*
(defun call-with-view (function
		       tensors
		       &key
			 (at-least-dim 1)
		       &aux
			 (shape (shape (car tensors)))
			 (dims  (length shape)))
  ;; Continue from here.

  (assert (every #'(lambda (tensor) (shape-equal-list (shape tensor) shape)) tensors)
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
				    (nth target-dim (shape tensor))
				    (let ((view (subscript-view (nth target-dim (tensor-view tensor)))))
				      (compute-stepby view))))))
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
	       (declare (ignorable axis-determined-p))
	       ;; When axis-determined-p is nil expand with loop for parts
	       ;; is t -> Unroll

	       (let ((stride-place (gensym "STRIDESTMP"))
		     (old-offsets offsets)
		     (loop-name (gensym "LOOP"))
		     (out (gensym "OUT"))
		     (offsets (gensym "Offsets")))
		 `(let ((,stride-place (list ,@(loop for tensor in tensors
						     collect (nth target-dim (tensor-stride tensor)))))
			(,offsets (copy-list ,old-offsets)))
		    (declare (type list ,stride-place))
		    ;; adds offset
		    ,(format nil "[NOTE] axis=~a ========" target-dim)
		    "[Note] ----- Adding First Offsets -----"
		    
		    ,@(loop for k upfrom 0
			    for tensor in tensors
			    collect `(setf (nth ,k ,offsets) (%+ (%* (nth ,k ,stride-place)
								     (%*
								      ,(compute-stepby (subscript-view (nth target-dim (tensor-view tensor))))
								      ,(nth k start-points)))
								 (nth ,k ,offsets))))

		    ,@(if (and axis-determined-p
			       (<= (car end-points) *unroll-threshold*))
			  (loop named iterator-loop
			        for nth fixnum upfrom 0 below (car end-points)
				collect `(progn
					   "[NOTE] Getting Next Round"
					   ,(if (<= rest-dim at-least-dim)
						(return-from iterator-loop (list (expand-with-function target-dim offsets)))
						(explore
						 (1- rest-dim)
						 offsets))
					   "[Note] Adding Offsets of this round"
					   ,@(unless (= nth (1- (car end-points)))
					       (expand-view-stride-adder
						nth
						offsets
						stride-place
						target-dim
						tensors))))
			  (let ((nth (gensym)))
			    `((loop named ,loop-name
				    for ,nth fixnum
				    upfrom 0
				      below ,(car end-points)
				    do
				    ,(progn
				       `(prog1
					    ,(if (<= rest-dim at-least-dim)
						 `(let ((,out ,(expand-with-function target-dim offsets)))
						    (return-from ,loop-name ,out))
						 (explore
						  (1- rest-dim)
						  offsets))
					  (unless (= ,nth (1- ,(car end-points)))
					    (progn
					      ,@(expand-view-stride-adder
						 nth
						 offsets
						 stride-place
						 target-dim
						 tensors))))))))))))))

    (let ((offset-place (gensym "OffsetsTmp"))
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



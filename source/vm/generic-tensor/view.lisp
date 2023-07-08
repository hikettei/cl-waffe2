
(in-package :cl-waffe2/vm.generic-tensor)

(defparameter *unroll-threshold* 3 "Unroll if iternum falls below this threshold")

;; TO Add: ViewInstruction2D to implement matmul
(defstruct (ViewInstruction
	    (:constructor
		make-viewinstruction (offset size by)))
  (size) ;; contains list, fixnum...
  (by)
  (offset))

(defun stride-of (views nth)
  (viewinstruction-by (nth nth views)))

(defun offset-of (views nth)
  (viewinstruction-offset (nth nth views)))

(defun size-of (views nth)
  (viewinstruction-size (nth nth views)))

(deftype subscript-t ()
  "An list of types which is allowed to become subscripts of the function view."
  `(or fixnum list null t))

(deftype subscript-syntax ()
  `(member :index :t :slice :slice-step :indices :tflist :broadcast))

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
			    (T (error "Unknown view keyword ~a, all available keywords are following: :indices :broadcast :repeat" view)))))

(defun compute-visible-actual-shape (tensor)
  (loop for v in (tensor-view tensor)
	for s in (slot-value tensor 'orig-shape)
	collect (- (compute-visible-end-idx-actual
		    (subscript-view v)
		    s)
		   (compute-visible-start-idx
		    (subscript-view v)))))

(declaim (ftype (function (subscript-t) fixnum) compute-visible-start-idx))
(defun compute-visible-start-idx (view)
  "Given view, this function returns a offset at the dimension."
  (declare (optimize (speed 3)))
  (read-symbol
   (case (viewtype view)
     (:index view)
     (:t     0)
     (:slice      (car view))
     (:slice-step (if (> (the fixnum (third view)) 0)
		      (min (the fixnum (car view)) (the fixnum (second view)))
		      (max (the fixnum (car view)) (the fixnum (second view)))))
     (:indices 0)
     (:tflist  0)
     (:broadcast 0)
     (T (error "unknown viewtype: ~a" view)))))

(declaim (ftype (function (subscript-t (or list symbol fixnum)) (or list symbol fixnum)) compute-visible-end-idx))
(defun compute-visible-end-idx (view size)
  "Given view and size, this function returns a size at the dimension."
  (read-symbol
   (case (viewtype view)
     (:index (1+ (the fixnum view)))
     (:t     size)
     (:slice (the fixnum (second view)))
     ;; FIXME: Should be divided :slice-step
     (:slice-step
      (the fixnum
	   (round (/ (if (> (third view) 0)
			 (max (car view) (second view))
			 (min (car view) (second view)))
		     (abs (third view))))))
     (:indices (length (cdr view)))
     (:tflist  size)
     (:broadcast (second view))
     (T (error "unknwon view: ~a" view)))))

(defun compute-visible-end-idx-actual (view size)
  (read-symbol
   (case (viewtype view)
     (:index (1+ view))
     (:t     size)
     (:slice (second view))
     (:slice-step
      (round (/ (if (> (third view) 0)
		    (max (car view) (second view))
		    (min (car view) (second view)))
		(abs (third view)))))
     (:indices (length (cdr view)))
     (:tflist  size)
     (:broadcast 1))))

(defun force-list (view)
  "Returns subscript-t if view is Subscript otherwise returns a view"
  (if (typep view 'subscript)
      (subscript-view view)
      view))

(defun compute-visible-shape (orig-shape view)
  "Returns a visible size of orig-shape given view."
  (loop for o in orig-shape
	for i upfrom 0
	collect (let* ((v     (or (nth i view) t))
		       (end   (compute-visible-end-idx   (force-list v) o))
		       (start (compute-visible-start-idx (force-list v))))
		  (cond
		    ((and (typep start 'fixnum)
			  (= start 0))
		     end)
		    ((and (typep end 'fixnum)
			  (typep start 'fixnum))
		     (abs (- end start)))
		    (T
		     `(abs (- ,end ,start)))))))

(defun actual-shape (tensor)
  (when (scalar-p tensor)
    (return-from actual-shape `(1)))
  
  (loop with view = (tensor-view tensor)
        for o in (original-shape tensor)
	for i upfrom 0
	collect (let* ((v     (or (nth i view) t))
		       (end   (compute-visible-end-idx-actual (force-list v) o))
		       (start (compute-visible-start-idx (force-list v))))
		  (cond
		    ((and (typep start 'fixnum)
			  (= start 0))
		     end)
		    ((and (typep end 'fixnum)
			  (typep start 'fixnum))
		     (abs (- end start)))
		    (T
		     `(abs (- ,end ,start)))))))

(defun parse-absolute (index size)
  "Returns a absolute index."
  (typecase size
    (fixnum (if (< index 0)
		(+ index size)
		index))
    (symbol
     (if (< index 0)
	 `(+ ,index ,size)
	 index))))


(defgeneric step-subscript (before-type after-type before after size))

;; ===================================================
;; [T] -> Any
;; ===================================================

(defmethod step-subscript ((x (eql :t))
			   (y (eql :index))
			   before
			   after
			   size)
  "Tensor[t][Index]
Return: (values after-view error)"

  ;; Parse -1, -2, -3...
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

    (if (> (car view) (second view))
	(progn
	  (setf (subscript-view after) `(,(second view) ,(car view) -1))
	  (step-subscript :t :slice-step before after size))
	;; -> step-subscript :t :slice-step when 10 -> 1
	(progn
	  (setf (subscript-char after) size
		(subscript-constraints after) `(,(second view) t)
		(subscript-view after) view)
	  after))))

(defmethod step-subscript ((x (eql :t))
			   (y (eql :slice-step))
			   before
			   after
			   size)
  "Tensor[t][2:4::-1]"
  (let* ((view (subscript-view after))
	 (view (map 'list #'(lambda (v) (parse-absolute v size)) (butlast view))))

    (if (> (car view) (second view))
	(progn
	  (setf (subscript-view after) `(,(second view) ,(car view) ,(- (third view))))
	  (step-subscript :t :slice-step before after size))
	(progn
	  (setf (subscript-char after) size
		(subscript-constraints after) `(,(max (car view) (second view)) t)
		(subscript-view after) `(,@view ,(third (subscript-view after))))
	  after))))

;; Ignore it for a while
(defmethod step-subscript ((x (eql :t))
			   (y (eql :indices))
			   before
			   after
			   size)
  
  "Tensor[t][:indices 1 2 3...]"
  (error "NOT IMPLEMENTED YET: <T> -> <INDICES>"))

;; Ignore it for a while
(defmethod step-subscript ((x (eql :t))
			   (y (eql :tflist))
			   before
			   after
			   size)
  
  "Tensor[t][:indices 1 2 3...]"
  (error "NOT IMPLEMENTED YET: <T> -> <TFLIST>"))

(defmethod step-subscript ((x (eql :t))
			   (y (eql :broadcast))
			   before
			   after
			   size)
  "Tensor[t][:broadcast n]"
  (setf (subscript-char after) size
	(subscript-constraints after) `(1 1))
  after)

;; ===================================================
;; Tensor[Index] -> Any
;; Tensor = (10, 10), Tensor[0] = (1, 10)
;; ===================================================

(defmethod step-subscript ((x (eql :index))
			   (y (eql :t))
			   before
			   after
			   size)
  "Tensor[0][t] is the same as just doing Tensor[t][0]"
  before)


(defmethod step-subscript ((x (eql :index))
			   (y (eql :index))
			   before
			   after
			   size)
"Tensor[0][1]"
  ;; Return error?
  ;; before == after
  (step-subscript :t :index before after size)
  )

(defmethod step-subscript ((x (eql :index))
			   (y (eql :slice))
			   before
			   after
			   size)
  "Tensor[0][0:10]"
  (error "Error: Index -> Slice")
  ;;(step-subscript :t :slice before after size)
  )

(defmethod step-subscript ((x (eql :index))
			   (y (eql :slice-step))
			   before
			   after
			   size)
  "Tensor[0][0:10::2]"
  (error "Error: Index -> Slice")
  ;;(step-subscript :t :slice-step before after size)
  )

(defmethod step-subscript ((x (eql :index))
			   (y (eql :indices))
			   before
			   after
			   size)
  "Tensor[1][:tflist t]"
  (error "Error: Index -> Indices"))

;; Ignore it for tmp
(defmethod step-subscript ((x (eql :index))
			   (y (eql :tflist))
			   before
			   after
			   size)
  "Tensor[1][:tflist t]"
  (error "Error: Index -> Indices"))

(defmethod step-subscript ((x (eql :index))
			   (y (eql :broadcast))
			   before
			   after
			   size)
  "Tensor[1][:broadcast 10]"
  ;; FIXME: Is offsets considered?
  ;; i.e.: Tensor[1][:broadcast 10] is working?
  (step-subscript :t :broadcast before after size))

;; ===================================================
;; Tensor[0:10] -> Any
;; ===================================================

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :t))
			   before
			   after
			   size)
  "Tensor[0:10][t] is the equivalent to Tensor[t][0:10]"
  before)

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :index))
			   before
			   after
			   size)
  "Tensor[2:10][0] Add offset to index.

I.e.: From viewpoint of x-orig, x-orig[1:5][0] is x-orig[1]"
  (incf (subscript-view after) (car (subscript-view before)))
  after)

(defmethod step-subscript ((x (eql :slice))
			   (y (eql :slice))
			   before
			   after
			   size)
  "Tensor[2:10][0:3]" ;; Just Adding Offsets
  (incf (car (subscript-view after))
	(car (subscript-view before)))
  (incf (second (subscript-view after))
	(car    (subscript-view before)))
  ;; (step-subscript :t :slice before after size)
  after)

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
  "Tensor[:broadcast 10][1] -> Tensor[:broadcast 1]"
  ;; TODO: ADD CONSTRAINTS
  ;; (setf (subscript-view after) 0)
  ;;(step-subscript :t :index before after size)
  (setf (subscript-view after) `(:broadcast 1))
  
  (step-subscript :t :broadcast
		  before
		  after
		  size))

(defmethod step-subscript ((x (eql :broadcast))
			   (y (eql :broadcast))
			   before
			   after
			   size)
  "Tensor[:broadcast 10][:broadcast 3]
Changes the number of broadcasting."
  (step-subscript :t :broadcast before after size))


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
		:determined-p determined-p ;; includes symbol?
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
  "
## [function] shape-equal

a=1, b=k => T
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


(declaim (ftype (function (subscript-t) fixnum) compute-stepby))
(defun compute-stepby (view)
  ;; view ... list
  (case (viewtype view)
    (:slice-step (third view))
    (:broadcast 0)
    (T 1)))

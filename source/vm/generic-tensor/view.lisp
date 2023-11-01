
(in-package :cl-waffe2/vm.generic-tensor)

;; Syntax:

;; - fixnum | `(:index ,value)
;; - (start stop)
;; - (start stop by) slice
;; - range
;; - T (all)
;; - `(:broadcast N)
;; - (~ FROM END STEP)
;; - ~  (Flexible)
;; - 1~ (Forced Flexible)

;; [TODO] 1~ = `(:broadcast T)として処理

;; [Optim] !sum produces a lot of compute-visible-start-idx
;;  This could be due to run-time creation of tensors?


(defun compute-visible-end-idx-actual (view size)
  (error "deleted")
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

(defun preprocess-subscript (dim tensor size past-view subscript)
  (declare (type fixnum dim)
	   (type AbstractTensor tensor)
	   (ignore dim))
  (error "deprecated")
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
  (error "deprecated")
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
      (= a b)
      (if (and (symbolp a)
	       (symbolp b))
	  (eql a b)
	  ;; Symbol + Number or Number + Symbol
	  t)))

(defun shape-equal-list (list1 list2)
  (every #'shape-equal list1 list2))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; 全部書き直す
;; tensor-view -> force-list -> tensor-view
;;  |- A list of range/bc ^ list  | A list of range
;;  復元できるようにする

(defstruct (ViewInstruction
	    (:constructor
		make-viewinstruction (offset size by)))
  "A structure indicating the way to call foreign function.
Utils:
  (stride-of views nth)
  (offset-of views nth)
  (size-of   views nth)"
  (size) ;; contains list, fixnum...
  (by)
  (offset))

(defun stride-of (views nth)
  (viewinstruction-by (nth nth views)))

(defun offset-of (views nth)
  (viewinstruction-offset (nth nth views)))

(defun size-of (views nth)
  (viewinstruction-size (nth nth views)))

(defstruct (Subscript
	    (:constructor make-subscript (view broadcast range)))
  (view view :type subscript-t)
  (broadcast broadcast :type boolean)
  (range range :type wf/iter:Range))

(defmethod print-object ((obj Subscript) stream)
  (format stream "<~a>" (subscript-view obj)))

(deftype subscript-t ()
  "An list of types which is allowed to become subscripts of the function view."
  `(or fixnum list null t cl-waffe2/vm.iterator:Range))

(deftype subscript-syntax ()
  `(member :index :t :slice :slice-step :broadcast :range))

(defun force-list (view)
  "Returns subscript if view is a subscript otherwise returns a given argument as it was"
  (if (typep view 'subscript)
      (subscript-view view)
      view))

(defmacro with-viewcase ((view)
			 &key
			   index
			   tcase
			   slice
			   keyword
			   range)
  `(cond
     ((and (listp ,view)
	   (= (length ,view) 2)
	   (or (eql (car ,view) :broadcast)
	       (eql (car ,view) :index)))
      ,keyword)
     ((and (listp ,view)
	   (or
	    (= (length ,view) 2)
	    (= (length ,view) 3)))
      ;; (from to) (from to by)
      ,slice)
     ((typep ,view 'cl-waffe2/vm.iterator:Range)
      ,range)
     (T
      (if (eql ,view t)
	  ,tcase
	  ,index))))

(declaim (ftype (function (subscript-t) subscript-syntax) viewtype))
(defun viewtype (view)
  (declare (optimize (speed 3))
	   (type subscript-t view))
  (with-viewcase (view)
		 :range :range
		 :index :index
		 :tcase :t
		 :slice (cond
			  ((= (length view) 2) :slice)
			  ((= (length view) 3) :slice-step)
			  (T (error "Cannot parse a view ~a
It should be (start stop) or (start stop step)"
				    view)))
		 :keyword (case (car view)
			    (:index :index)
			    (:broadcast :broadcast)
			    (T (error "Unknown view keyword ~a, all available keywords are following: :index :broadcast" view)))))

(defun view->range (length view)
  (case (viewtype view)
    (:index      (wf/iter:range view))
    (:slice      (apply #'wf/iter:range view))
    (:slice-step (apply #'wf/iter:range view))
    (:t          (wf/iter:range 0 length))
    (:range      view)
    (:broadcast  (wf/iter:range length))))

(defun compute-next-view (tensor past-view subscripts)
  "
Creates a view for thegiven tensor.
Return: List[Subscript]
"
  (loop for base-size in (slot-value tensor 'orig-shape)
	for N fixnum upfrom 0
	collect
	;; In default, T
	(parse-view base-size
		    (when (nth N past-view)
		      (wf/iter:range-size
		       (subscript-range (nth n past-view))))
		    (force-list (nth N past-view))
		    (force-list (or (nth N subscripts) T)))))

(defun parse-view (base-size viewed-shape past-view view)
  (let* ((latest-view (view->range (or viewed-shape base-size) (force-list view)))
	 (old-view    (when past-view
			(view->range base-size (force-list past-view))))
	 (composed-range (wf/iter:.range latest-view old-view)))
    (make-subscript
     view
     (and (listp (force-list view))
	  (eql (car (force-list view)) :broadcast))
     composed-range)))

(declaim (ftype (function (subscript-t) fixnum) compute-stepby))
(defun compute-stepby (view)
  ;; view ... list
  (case (viewtype view)
    ;;(:slice-step (wf/iter:range-step (apply #'wf/iter:range view)))
    (:broadcast 0)
    (T 1)))

(defun compute-visible-shape (tensor)
  (loop for v in (tensor-view tensor)
	collect (wf/iter:range-size (subscript-range v))))

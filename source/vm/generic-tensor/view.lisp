
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
	  ;; Proceeds with asserting that A is determined to the symbol B.
	  (progn
	    (wf/vm::lazy-assert A B)
	    t)
	  ;; Proceeds with asserting that A is determined to number B.
	  (progn
	    (wf/vm::lazy-assert A B)
	    t))))

(defun shape-equal-list (list1 list2)
  (every #'shape-equal list1 list2))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
  (ecase (viewtype view)
    (:index      (wf/iter:range view))
    (:slice      (apply #'wf/iter:range view))
    (:slice-step (apply #'wf/iter:range view))
    (:t          (wf/iter:range 0 length))
    (:range      view)
    (:broadcast  (wf/iter:range 0))))

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

(defun parse-absolute (size subscript)
  (if (and (numberp subscript)
	   (< subscript 0))
      (cl-waffe2/vm:make-lazyaxis `(+ ,size ,subscript))
      subscript))

(defun parse-view (base-size viewed-shape past-view view)
  (let* ((view (case (viewtype view)
		 (:index
		  (parse-absolute viewed-shape view))
		 (:slice
		  (list
		   (parse-absolute viewed-shape (car view))
		   (parse-absolute viewed-shape (second view))))
		 (:slice-step
		  (list
		   (parse-absolute viewed-shape (car view))
		   (parse-absolute viewed-shape (second view))
		   (third view)))
		 (T
		  view)))		   
	 (latest-view (view->range (or viewed-shape base-size) (force-list view)))
	 (old-view    (when past-view
			(view->range base-size (force-list past-view))))
	 (composed-range (wf/iter:.range latest-view old-view)))
    
    ;; <T> to try to extend the previous view.
    (or
     (when (and old-view (eql view t))
       (if (and (listp (force-list old-view))
		(eql (car (force-list old-view)) :broadcast))
	   (make-subscript
	    past-view
	    nil
	    old-view)
	   (make-subscript
	    view
	    nil
	    (wf/iter:range 0 base-size))))
     (make-subscript
      view
      (and (listp   (force-list view))
	   (eql (car (force-list view)) :broadcast))
      composed-range))))

(declaim (ftype (function (subscript-t) fixnum) compute-stepby))
(defun compute-stepby (view)
  ;; view ... list
  (case (viewtype view)
    ;;(:slice-step (wf/iter:range-step (apply #'wf/iter:range view)))
    (:broadcast 0)
    (T 1)))

(defun compute-visible-shape (tensor)
  (loop for v in (tensor-view tensor)
	if (subscript-broadcast v)
	  collect (second (force-list v))
	else
	  collect (wf/iter:range-size (subscript-range v))))

(defun actual-shape (tensor)
  "
## [function] actual-shape

Computes a shape but broadcasted axis is replaced with 1.
"
  (when (scalar-p tensor)
    (return-from actual-shape `(1)))
  
  (loop for s in (shape tensor)
	for v in (tensor-view tensor)
	if (subscript-broadcast v)
	  collect 1
	else
	  collect s))


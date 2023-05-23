
(in-package :cl-waffe2/vm.generic-tensor)

;; These a ton of awful codes are from cl-xmatrix.
(deftype subscript-t ()
  "An list of types which is allowed to become subscripts of the function view."
  `(or fixnum list null t))

(deftype index () `(or fixnum))

(declaim (ftype (function (fixnum subscript-t fixnum) list) find-subscript-error))
(defun find-subscript-error (i sub dim &aux (reports nil))
  "Finding view-indexing-error in advance.
Args:
- i     the axis
- sub   subscripts[axis]
- dim   visible-shape[axis]

Return: List (Consisted of strings which records error log)"
  (declare (optimize (speed 3) (safety 0))
	   (type fixnum i dim)
	   (type subscript-t sub))
  (typecase sub
    (fixnum
     (if (< (the fixnum sub) (the fixnum dim))
	 t
	 (push
	  (format nil "[Axis=~a] Failed with (< subscript[~a] shape[~a]). subscript[~a]=~a shape[~a]=~a~%"
		  i
		  i
		  i
		  i
		  sub
		  i
		  dim)
	  reports)))
    (list
     (typecase (car sub)
       (fixnum
	(cond
	  ;; (5 3) is invaild (>= 5 3)
	  ((>= (the fixnum (car sub)) (the fixnum (second sub)))
	   (push
	    (format nil "[Axis=~a] Failed with (< subscript[~a][0] subscript[~a][1]). subscript=~a~%"
		    i
		    i
		    i
		    sub)
	    reports))
	  ;; (n 10) but axis=3
	  ((> (the fixnum (second sub)) (the fixnum dim))
	   (push
	    (format nil "[Axis=~a] Failed with (< subscript[~a][1] shape[~a]) subscript=~a, shape[~a]=~a~%"
		    i
		    i
		    i
		    sub
		    i
		    dim)
	    reports))
	  ((not (= (length sub) 2))
	   (push
	    (format nil "[Axis=~a] Failed with (= (length subscript[~a]) 2). subscript=~a.~%"
		    i
		    i
		    sub)
	    reports))
	  (t t)))
       (keyword
	(case (car sub)
	  (:indices
	   (let ((ov (find-if #'(lambda (x) (>= (the fixnum x) dim)) (the list (cdr sub)))))
	     (if ov
		 (push
		  (format nil "[Axis=~a] Each index mustn't exceed ~a, but found: ~a.~%"
			  i
			  dim
			  ov)
		  reports)
		 t)))
	  (:broadcast
	   (cond
	     ((not (= (length sub) 2))
	      (push
	       (format nil "[Axis=~a] :broadcast keyword is given the following format: `(:broadcast n) but got ~a~%"
		       i
		       sub)
	       reports))
	     ((not (= (the fixnum dim) 1))
	      (push
	       (format nil "[Axis=~a] The axis to be broadcasted, must be 1 but got ~a.~%" i dim)
	       reports))))
	  (:tflist
	   ;; add type checks
	   t)
	  (T
	   (push
	    (format nil "[Axis=~a] Unknown keyword: ~a~%" i sub)
	    reports))))))
    (t
     (if (eql sub t)
	 t
	 (push
	  (format nil "[Axis=~a] Invaild argument ~a~%" i sub)
	  reports))))
  reports)


(defun compute-absolute-subscript (old-view subscript)
  "Translate view-subscription into the format which is compatiable with orig-mat"
  (declare (optimize (speed 3) (safety 0))
	   (type subscript-t old-view subscript))

  (labels ((handle-ext-index (view sub)
	     ;; note: don't return sub directly, add view.
	     (typecase view
	       (index
		;; M[2][0]
		(the index (+ view (the index sub))))
	       (list
		(typecase (car view)
		  (keyword
		   ;; M[:broadcast 10][1]
		   ;; M[:indices 1 2 3 4][1]
		   (case (car view)
		     (:indices
		      (nth sub (cdr view)))
		     (:broadcast 0)))
		  (index
		   ;; M[2:4].view(1)
		   (the index
			(+ (the index (car view)) (the index sub))))
		  (T
		   (view-indexing-error "Can't handle with this instruction: ~a" view))))
	       (t
		;; M[T][0]
		sub)))
	   (handle-ext-range (view sub)
	     (typecase view
	       (index
		;; M[1].view([2:4])
		(view-indexing-error "Attempted to comptute Matrix[~a][~a] but this is out of range." view sub))
	       (list
		(typecase (car view)
		  (keyword
		   ;; M[:broadcast 10][0:2]
		   ;; M[:indices 1 2 3 4][0:2]
		   ;; Todo: Detect out of range
		   (case (car view)
		     (:broadcast
		      `(:indices
			,@(loop for i fixnum upfrom (car sub) below (second sub)
				collect 0)))
		     (:indices
		      `(:indices
			,@(loop for i fixnum upfrom (car sub) below (second sub)
				collect (nth i (cdr view)))))))
		  (index
		   ;; M[2:10][1:2] -> M[3:5]
		   ;; Todo: Detect out of range.
		   `(,(the index (+ (the index (car view)) (the index (car sub))))
		     ,(the index (+ (the index (car view)) (the index (second sub))))))
		  (T
		   (view-indexing-error "Cant handle this subscript: ~a" view))))
	       (t sub)))

	   (handle-ext-kw (view sub)
	     (typecase view
	       (index
		;; M[0][:indices 1 2 3]
		(view-indexing-error "Attempted to compute M[~a][~a] but it is out of range." view sub))
	       (list
		(typecase (car view)
		  (keyword
		   ;; M[:broadcast 10][:indices 0 1]
		   ;; M[:indices 1 2 3][:indices 0 1]
		   (case (car view)
		     (:broadcast
		      `(:indices ,@(loop for i upfrom 0 below (the index (second sub)) collect 0)))
		     (:indices
		      `(:indices ,@(map
				    'list
				    #'(lambda (k)
					(nth k (cdr view)))
				    (cdr sub))))))
		  (index
		   ;; M[2:6][:indices 1 2]
		   (let ((ls (loop for i fixnum
				   upfrom (car view)
				     below (second view)
				   collect i)))
		     `(:indices ,@(map
				   'list
				   #'(lambda (k)
				       (nth k ls))
				   (cdr sub)))))
		  (T
		   (view-indexing-error "Cant handle this subscript: ~a" view))))
	       ;; M[T][:indices 1 2 3]
	       (t sub)))
	   (handle-ext-kw-broadcast (view sub)
	     (typecase view
	       (index
		;; M[0][:broadcast 10]
		sub)
	       (list
		(typecase (car view)
		  (keyword
		   ;; M[:indices 0 1 2 3][:broadcast 10]
		   ;; M[:broadcast 10][:broadcast 10]
		   (if (and (= (length (the list (cdr view))) 1)
			    (eql (car view) :indices))
		       ;; Only after [:indices m]
		       sub
		       (view-indexing-error "view: ~a and ~a couldn't broadcasted together. The axis to be broadcasted, must be 1. Also, M[:broadcast 1][:broadcast 1] is prohibited. (Make a copy of matrix and try it again plz.)" view sub)))
		  (index
		   ;; M[0:10][:broadcast 10]
		   (if (= (- (the index (second view)) (the index (car view))) 1)
		       sub
		       (view-indexing-error "view: ~a and ~a couldn't broadcasted together. The axis to be broadcasted, must be 1." view sub)))
		  (T
		   (view-indexing-error "Cant handle this subscript: ~a" view))))
	       ;; M[T][:indices 1 2 3]
	       (t sub))))
    (typecase subscript
      (index (handle-ext-index old-view subscript))
      (list
       (typecase (car subscript)
	 (keyword
	  (case (car subscript)
	    (:broadcast (handle-ext-kw-broadcast old-view subscript))
	    (:indices   (handle-ext-kw           old-view subscript))))
	 (fixnum
	  (handle-ext-range old-view subscript))
	 (t (view-indexing-error "Cannot handle with this subscript: ~a" subscript))))
      (t old-view))))


(declaim (ftype (function (fixnum subscript-t AbstractTensor fixnum) (values subscript-t subscript-t)) parse-broadcast))
(defun parse-broadcast (orig-shape subscript tensor axis)
  "If subscript is broadcast. Returns the broadcasted shape and new-subscript. do nothing otherwise."
  (declare (optimize (speed 3) (safety 0)) 
	   (type fixnum orig-shape axis)
	   (type subscript-t subscript)
	   (type AbstractTensor tensor))

  (if (and (typep subscript 'list)
	   (eql (car subscript) :broadcast))
      (progn
	(unless (= (length subscript) 2)
	  (view-indexing-error "Invaild Operation ~a. :broadcast is given in this format:~% `(:broadcast num) num = t or positive fixnum" subscript))
	(unless (= orig-shape 1)
	  (view-indexing-error "Can't Broadcast the matrix~%because the axis to be broadcasted is not 1: ~a at axis=~a" (shape tensor) axis))
	(values t (second subscript)))
      (values subscript nil)))

(declaim (ftype (function (fixnum subscript-t) subscript-t) replace-tflist)
	 (inline replace-tflist))
(defun replace-tflist (orig-shape subscript)
  "Replace :tflist into :indices if exists, otherwise do nothing."
  (declare (optimize (speed 3))
	   (type fixnum orig-shape)
	   (type subscript-t subscript)
	   (ignore orig-shape))

  (if (and (typep subscript 'list)
	   (eql (car subscript) :tflist))
      (error "Currently AbstractTensor doesnt suport :tflist (TODO)")
      subscript))

(declaim (ftype (function (subscript-t) (values subscript-t subscript-t))
		parse-external-operation)
	 (inline parse-external-operation))
(defun parse-external-operation (subscript)
  "Parses :indices if exists otherwise do nothing."
  (declare (type subscript-t subscript)
	   (optimize (speed 3) (safety 0)))
  (if (and (typep subscript 'list)
	   (eql (car subscript) :indices))
      (values subscript subscript)
      (values subscript nil)))


(declaim (ftype (function (fixnum subscript-t) subscript-t)))
(defun parse-relative-position (orig-shape sub)
  "Parses relatie-position like: -1, -2...
specifing :- means orig-shape (todo: write docs)"
  (declare (optimize (speed 3) (safety 0))
	   (type fixnum orig-shape)
	   (type subscript-t sub))
  (typecase sub
    (fixnum
     (if (>= sub 0)
	 sub
	 (let ((pos (the fixnum (+ orig-shape sub))))
	   (if (>= pos 0)
	       pos
	       (view-indexing-error "The relative index ~a beyonds the original axis ~a" sub orig-shape)))))
    (list
     (map 'list #'(lambda (x) (parse-relative-position orig-shape x)) sub))
    (T (if (eq sub :~)
	   orig-shape
	   sub))))

(declaim (ftype (function (subscript-t subscript-t) fixnum) compute-visible-size))
(defun compute-visible-size (shape view)
  (declare (optimize (speed 3) (safety 0)))
  (the fixnum
       (- (view-endindex view shape)
	  (view-startindex view 0))))

(declaim (ftype (function (t fixnum) fixnum) view-startindex view-endindex))
(defun view-startindex (view _)
  (declare (optimize (speed 3) (safety 0))
	   (ignore _))
  (typecase view
    (list
     (typecase (car view)
       (fixnum (the fixnum (car view)))
       (keyword
	(case (car view)
	  (:indices 0)
	  (:broadcast 0)
	  (T
	   (error "view-startindex: unknown keyword"))))
       (T (error "view-startindex: invaild view-instruction fell through"))))
    (fixnum
     (the fixnum view))
    (t
     (the fixnum 0))))

(defun view-endindex (view shape)
  (declare (optimize (speed 3) (safety 0)))
  (typecase view
    (list
     (typecase (car view)
       (fixnum (the fixnum (second view)))
       (keyword
	(case (car view)
	  (:indices (1- (length view)))
	  (:broadcast 1)
	  (T
	   (error "view-endindex: unknown keyword"))))
       (T (error "view-endindex: unknown view-instruction fell through"))))
    (fixnum
     (the fixnum (1+ view)))
    (t
     (the fixnum shape))))



(defmacro unroll-maplist ((var iter-num) &body body)
  (labels ((mkstr (&rest args)
	     (with-output-to-string (s)
	       (dolist (a args) (princ a s))))
	   
	   (symb (&rest args)
	     (values (intern (apply #'mkstr args))))
	   
	   (retain-objects (name i)
	     `(list ,@(loop for k fixnum upfrom 0 below i
			    collect (symb name k))))
	   (step-iter (i)
	     (if (>= i 0)
		 `(multiple-value-bind
			(,(symb 'subscript i)
			 ,(symb 'broadcast i)
			 ,(symb 'visible-shape i)
			 ,(symb 'external-operation i)
			 ,(symb 'external-operation-dim i)
			 ,(symb 'error-str i))
		      (let ((,var ,i)) ,@body)
		    ,(step-iter (1- i)))
		 `(values
		   ,(retain-objects 'subscript iter-num)
		   ,(retain-objects 'broadcast iter-num)
		   ,(retain-objects 'visible-shape iter-num)
		   ,(retain-objects 'external-operation iter-num)
		   ,(retain-objects 'external-operation-dim iter-num)
		   ,(retain-objects 'error-str iter-num)))))
    (step-iter (1- iter-num))))

;; TODO: :tflist  etc...


(declaim (ftype (function (fixnum
			   AbstractTensor
			   fixnum
			   (or fixnum list null t)
			   (or fixnum list null t)
			   boolean)
			  (values t t t t t t))
		parse-subscript-by-axis))
(defun parse-subscript-by-axis (axis
				tensor
				orig-shape
				orig-view
				subscript
				padding-subscript)
  "Returning -> (values parsed-subscript[sub] broadcast[Fixnum] visible-shape[fixnum] external-operation[null or list]) external-operation-dim-num[fixnum] Errors[null or string]"
  (declare (optimize (speed 3) (safety 0))
	   (type fixnum orig-shape)
	   (type boolean padding-subscript)
	   (type (or fixnum list t) orig-view subscript))
  ;; Works:
  ;; Detect View error
  ;; compute visible-shape
  ;; compute absolute view
  ;; Separate Broadcasting
  ;; compute ext-ops (:indices) (:tflist)

  ;; Here, Make -1 -> 1 (Compute Absolute)
  ;; If :tflist, convert it into :indices
  
  (let* ((subscript (parse-relative-position orig-shape subscript))
	 (subscript (replace-tflist orig-shape subscript)) ;; :tflist -> :indices
	 (subscript (if padding-subscript ;; Padding
			t
			subscript))
	 (subscript (if (null subscript)
			t
			subscript)))
    (multiple-value-bind (subscript broadcast) (parse-broadcast orig-shape subscript tensor axis) ;; Find out broadcasts
      (multiple-value-bind (subscript external-operation) (parse-external-operation subscript) ;; Find out :indices
	(if (tensor-projected-p tensor)
	    (let ((subscript (compute-absolute-subscript orig-view subscript)))
	      (values
	       subscript
	       broadcast
	       (or broadcast
		   (compute-visible-size orig-shape subscript))
	       external-operation
	       (if external-operation
		   axis)
	       ;; Error check will be done in: Original-Mat <-> View
	       (find-subscript-error axis subscript (nth axis (slot-value tensor 'orig-shape)))))
	    ;; Simply, matrix -> View
	    (values
	     subscript
	     broadcast
	     (or broadcast
		 (compute-visible-size orig-shape subscript))
	     external-operation
	     (if external-operation
		 axis)
	     (find-subscript-error axis subscript orig-shape)))))))

(defun parse-view-subscripts (tensor
			      subscripts
			      &aux
				(orig-shape (slot-value tensor 'orig-shape))
				(orig-view  (slot-value tensor 'view))
				(dimensions (length orig-shape))
				(subscript-len (length subscripts)))
  (declare (optimize (speed 3) (safety 0))
	   (type AbstractTensor tensor)
	   (type list subscripts orig-shape orig-view)
	   (type fixnum dimensions))
  ;; Assertion: (100% as long as created by matrix) (length orig-shape) == (length orig-view)
  
  (unless (>= dimensions subscript-len)
    (view-indexing-error
     "The length of subscripts is too large for the given matrix.~%Matrix:     ~a~%Subscripts: ~a"
     (shape tensor) subscripts))

  ;; SBCL's IRC Bug: Unsafe concurrent operations on #<HASH-TABLE :TEST EQL :COUNT 14 {100D1E9F03}> detected, the following ugly case clauses below can't be rewritten with more briefly notations/macros ;_;.
  (case dimensions
    (1
     (unroll-maplist (i 1)
       (parse-subscript-by-axis
	i
        tensor
	(nth i orig-shape)
	(nth i orig-view)
	(nth i subscripts)
	(>= i subscript-len))))
    (2
     (unroll-maplist (i 2)
       (parse-subscript-by-axis
	i
	tensor
	(nth i orig-shape)
	(nth i orig-view)
	(nth i subscripts)
	(>= i subscript-len))))
    (3
     (unroll-maplist (i 3)
       (parse-subscript-by-axis
	i
	tensor
	(nth i orig-shape)
	(nth i orig-view)
	(nth i subscripts)
	(>= i subscript-len))))
    (4
     (unroll-maplist (i 4)
       (parse-subscript-by-axis
	i
	tensor
	(nth i orig-shape)
	(nth i orig-view)
	(nth i subscripts)
	(>= i subscript-len))))
    (5
     (unroll-maplist (i 5)
       (parse-subscript-by-axis
	i
	tensor
	(nth i orig-shape)
	(nth i orig-view)
	(nth i subscripts)
	(>= i subscript-len))))
    (6
     (unroll-maplist (i 6)
       (parse-subscript-by-axis
	i
	tensor
	(nth i orig-shape)
	(nth i orig-view)
	(nth i subscripts)
	(>= i subscript-len))))
    (7
     (unroll-maplist (i 7)
       (parse-subscript-by-axis
	i
	tensor
	(nth i orig-shape)
	(nth i orig-view)
	(nth i subscripts)
	(>= i subscript-len))))
    (8
     (unroll-maplist (i 8)
       (parse-subscript-by-axis
	i
	tensor
	(nth i orig-shape)
	(nth i orig-view)
	(nth i subscripts)
	(>= i subscript-len))))
    (T
     (let ((subs)
	   (bcs)
	   (vshapes)
	   (ext-opes)
	   (ext-dims)
	   (error-list))
       (dotimes (i dimensions)
	 (multiple-value-bind
	       (subscript
		broadcast
		visible-shape
		external-operation
		ext-ope-num
		err)
	     (parse-subscript-by-axis i tensor (nth i orig-shape) (nth i orig-view) (nth i subscripts) (>= i subscript-len))
	   (push subscript subs)
	   (push broadcast bcs)
	   (push visible-shape vshapes)
	   (push external-operation ext-opes)
	   (push ext-ope-num ext-dims)
	   (push err error-list)))
       (values
	(reverse subs)
	(reverse bcs)
	(reverse vshapes)
	(reverse ext-opes)
	(reverse ext-dims)
	(reverse error-list))))))


;; (defun with-view
;; Displacementを調整しながらoffsets, m, nなどを調整したIndexを計算



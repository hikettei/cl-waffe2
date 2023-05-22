
(in-package :cl-waffe2/vm.generic-tensor)

;; tensors must support displacement.

;; cl-xmatrixからViewを移植する.
;; displacementの調整とUnrollだけでBroadcastingなどを実現する

(deftype subscript-t ()
  "An list of types which is allowed to become subscripts of the function view."
  `(or fixnum list null t))

(deftype index () `(or fixnum))

(declaim (ftype (function (fixnum subscript-t fixnum) list) find-subscript-error)
	 (inline find-subscript-error))
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


;; TODO: :tflist  etc...


(in-package :cl-waffe2/vm.iterator)

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Polyhedral Compiler
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; T=0 |for(i=0..2) } => Timestamp(0, i)   \
;;  I=1| X[1] += 1; } =>  |Timestamp(0, 1)  | => f(Timestamp(0, i)) -> New_Timestamp(...) where f = Affine Transformation.
;;  I=2| X[2] += 2; } =>  |Timestamp(0, 2) /
;; T=1 |for(i=0..2) } => Timestamp(1, i)   \
;;  I=1| X[1] += 1; } =>  |Timestamp(1, 1) | => f(Timestamp(1, i)) -> New_Timestamp(...) where f = Affine Transformation.
;;  I=2| X[2] += 2; } =>  |Timestamp(1, 2) /
;;
;; Timestamp is a structure comprised of a list of index referencing:
;;  e.g.: (i, j) = (0, 0), (0, 1) ... where (i<=10), (j<=10)
;;

;; The goal of this file is to find the best route to achive the best performance
;;  Procedure e.g.: :rotate -> :shuffle -> :fuse ...

;; Iteration_Space
;;  i - A[i, j]
;;  |         X X
;;  |       X X X
;;  |     X X X X
;;  |   X X X X X
;;  | X X X X X X
;;  |------------- j
;; => find out the combination of (i, j) * f minimizing the loss.

(defstruct (Timestamp
	    (:conc-name ts-)
	    (:constructor make-timestamp (indices constraints)))
  "Indices     = a list of iterators e.g.: (i, j, k)
   Constraints = a list of maxsize of eatch indices
   Transform   = a list of lambda function which represents the transformation"
  (rank (length indices) :type fixnum)
  (constraints constraints :type list)
  (transforms nil :type list))

;; TODO: Optimize
(defun apply-schedule (timestamp scale &optional bias)
  (declare (type Timestamp timestamp))
  (let* ((scale  (numcl:asarray scale :type 'fixnum))
	 (bias   (when bias (numcl:asarray bias :type 'fixnum))))
    (push
     #'(lambda (i)
	 (declare (type (array fixnum (* * *)) i)
		  (type (array fixnum (* *)) scale))
	 (let ((z (numcl:einsum '(ij bjk -> bik) scale i)))
	   (declare (type (array fixnum (* * *)) z))
	   (if bias
	       (locally
		   (declare (type (array fixnum (*)) bias))
		 (numcl:+ z bias))
	       z)))
     (ts-transforms timestamp))
    timestamp))

(defun realize (timestamp)
  (declare (type Timestamp timestamp))
  (let* ((coords
	   (apply
	    #'alexandria:map-product
	    #'list
	    (mapcar #'alexandria:iota (ts-constraints timestamp))))
	 (dims `(,(apply #'* (ts-constraints timestamp)) ,(ts-rank timestamp)))
	 (in (make-array
	      dims
	      :element-type 'fixnum
	      :initial-contents
	      coords))
	 (in (numcl:asarray in :type 'fixnum))
	 ;; (60, 1, 3) e.g.:
	 (in (numcl:reshape in `(,(first dims) ,(second dims) 1))))
    (loop with gained = in
	  for transform in (reverse (ts-transforms timestamp)) do
	    (setf gained (funcall transform gained))
	  finally (return-from realize gained))))

;;
;; Constraints
;; 1. Ranks are the same, upper/lower bounds are determined.
;; 2. Iterations must be increased by one.
;;

(defun all-permutations (list)
  (cond ((null list) nil)
        ((null (cdr list)) (list list))
        (t (loop for element in list
		 append (mapcar (lambda (l) (cons element l))
				(all-permutations (remove element list)))))))

(defun polyhedral-optimize! (schedule)
  (declare (type Scheduler schedule))

  )


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

;;
;; Constraints
;; 1. Ranks are the same, upper/lower bounds are determined.
;; 2. Iterations must be increased by one.
;;

;; Generalized definition of S(vec_i) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; [Unrolled Table]
;; S(i j k) =
;; (0 0 0)      }
;; (0 0 1)      }
;;   ...        }
;; (0 0 k-1)    }
;; (0 0 k)      }
;; (0 1 0)      }
;; (0 1 1)      }
;;   ...        }
;; (0 1 k-1)    }
;; (0 1 k)      }
;;   ...        }
;; (0 j-1 k-1)  }
;;  ...         }
;; (i j k)      } Shuffling the order of execution; there could be better one which parallelizes it or maximizes the locality of memory.
;;  | \  \_
;; ~~~~~~~~~~~~~~
;;  i  j  k

;; e.g.: out[ik] = out[ik] + x[ij] * y[jk]
;;  in the dimension=0,                                                
;;    - out[i] reads out[i], x[i], y[j]                                
;;      - that is, y[j] must be computed at that time.                 
;;      - <=> In the unrolled table, the transformed order satisfies:
;;            - | T=n-k | finish computing y[j]   } 
;;            - | T=n   | finish computing out[i] }
;;            - Polyhedral Dependence: Forall i, j. coeff1 * i + offset1 > coeff2 * j + offset2
;;
;;  in the dimension=1,
;;    - out[k] reads out[k], x[j], y[k]
;;      - that is, x[j] must be computed at that time.
;;      - <=> In the unrolled table, the transformed order satisfied:
;;            - | as well as dimension=0 |
;;            - Polyhedral Dependence: Forall k, j. coeff1 * k + offset1 > coeff2 * j + offset2
;;
;; dimension=0 and dimension=1 is independent; like the time on a clock.
;;

;; (C_11 C_12 C_13)
;;       scale                        offset
;; ( C_xy ... C_xy ) }                (C_x)
;;        ...        } d @ S(vec_i) + (C_x) d = S_new(vec_i)
;; ( C_xy ... C_xy ) }                (C_x)
;;         m                            1

;; [TODO]: Somebody help me with understanding and improving this stupid implementation T_T.
(defun polyhedral-optimize (schedule n-threads
			     &aux
			       (sorted (sort-stage schedule)))
  (declare (type Scheduler schedule))
  ;; Finds out the best scale
  (flet ((C (&rest indices) (apply #'symb 'C (loop for i in indices append `(- ,i)))) ;; Coeff
	 (lazy-matmul-helper (A B n-rank
			      &aux (out (make-array `(,n-rank) :element-type 'symbol :initial-element nil)))
	   (dotimes (i n-rank)
	     (dotimes (j n-rank)
	       ;; ij, jk -> ik
	       (let ((read-out (aref out i)))
		 (setf (aref out i)
		       `(,@read-out (* ,(aref A i j) ,(nth j B)))))))
	   out)
	 (linearlize-subject (s1 s2 c1 c2 offset1 offset2 p0 p1 &aux (b (- offset1 offset2)))
	   ;; L_n = (s_n - c_n)
	   ;; b = offset1 - offset2
	   ;; L_n * i +  L_n * j + L_N * k + ... + b >= 0 ... (1)
	   ;; <=> (Farkas Lemma)
	   ;;  exists p0, p1 >= 0, (1) is the equivalent to satisfy:
	   ;;  - p0 >= 0 and p1 >= 0 and s1c1 - s2c2 - p1 = 0	   
	   `((>= ,p0 ,b)
	     (>= ,p1 1)
	     (=
	      (-
	       (* ,s1 ,c1)
	       (* ,s2 ,c2)
	       ,p1)
	      0)))
	 (make-indices (&aux (sorted (alexandria:flatten sorted)))
	   (let ((indices
		   (delete-duplicates
		    (map
		     'list
		     #'iterstage-determines
		     sorted)))
		 (table (make-hash-table)))
	     (dolist (i indices)
	       (setf (gethash i table)
		     (iterstage-size (find i sorted :key #'iterstage-determines :test #'eql))))
	     (values indices table)))
	 (actions (&aux (sorted (alexandria:flatten sorted)))
	   (loop for s in sorted
		 for i = (iterstage-ops s)
		 append i)))
    (let* ((constraints (multiple-value-list (make-indices)))
	   (indices     (first  constraints))
	   (table       (second constraints))
	   (scale-to-minimize (make-array
			       `(,(length indices) ,(length indices))
			       :element-type 'list
			       :initial-contents
			       (loop for x upfrom 0 below (length indices)
				     collect
				     (loop for y upfrom 0 below (length indices)
					   collect (C x y)))))
	   (constraints-on-c
	     (loop for x upfrom 0 below (length indices)
		   append
		   (loop for y upfrom 0 below (length indices)
			 append
			 `((integer ,(C x y))
			   (>= ,(C x y) 0)
			   (<= ,(C x y) 1)))))
	   (constraints-on-c1
	     (loop for x upfrom 0 below (length indices)
		   collect
		   `(=
		     1
		     (+
		      ,@(loop for y upfrom 0 below (length indices)
			      collect (C x y))))))
	   (iterator-constraints
	     (loop for index in indices
		   append
		   `((>= ,index 0)
		     (<=  ,index ,(1- (gethash index table)))
		     (integer ,index))))
	   (rotated-schedule (lazy-matmul-helper scale-to-minimize indices (length indices)))
	   (i2r
	     (let ((table (make-hash-table)))
	       (loop for k in indices
		     for val across rotated-schedule do
		       (setf (gethash k table) val))
	       table))
	   (objects)
	   (schedule-applied-indices
	     (loop for action in (actions)
		   append
		   (loop for reader in (action-source action)
			 for r-t = (ispace-tensor reader)
			 for r-s = (ispace-space reader)
			 append
			 ;; target <- reader
			 (loop for writer in (action-target action)
			       for w-t = (ispace-tensor writer)
			       for w-s = (ispace-space writer)
			       append
			       (progn
				 (assert (= (dims r-t) (dims w-t))
					 ()
					 "Assertion Failed with (dims r-t) == (dims w-t) This could be an internal bug of polyhedral compiler.")
				 (loop for reader-dim in r-s
				       for writer-dim in w-s
				       for nth upfrom 0
				       unless (and
					       (eql (iref-index writer-dim) (iref-index reader-dim)))
					 
					 append
				       ;; Farkas Lemma:
				       ;; DAG: Writer <- Reader
				       ;; <=> Writer Depends Reader
				       ;; <=> Reader sends a data to Writer
				       ;; <=> for all ReaderIndex <= WriterIndex
					 (let ((wd (second (nth nth (gethash (iref-index writer-dim) i2r))))
					       (rd (second (nth nth (gethash (iref-index reader-dim) i2r))))
					       (p0 (gensym "p0"))
					       (p1 (gensym "p1")))
					   ;; Minimize the stride of reading values:
					   (when (= (1- (length r-s)) nth)
					     (push
					      `(* ,(iref-stride writer-dim) ,wd)
					      objects)
					     (push
					      `(* ,(iref-stride reader-dim) ,rd)
					      objects))
					   ;; Reader comes the first <-> Reader is smaller
					   ;; {ax+b} - {ax+b} >= 0
					   ;;`(>=
					   ;;  (+ (* ,wd ,(iref-stride writer-dim)) ,(iref-offset writer-dim))
					   ;;  (+ (* ,rd ,(iref-stride reader-dim)) ,(iref-offset reader-dim)))					   
					   ;; <=>
					   (linearlize-subject
					    (iref-stride writer-dim)
					    (iref-stride reader-dim)
					    wd ;; c
					    rd ;; c
					    (iref-offset writer-dim)
					    (iref-offset reader-dim)
					    p0
					    p1))))))))
	   (objective
	     ;; Object: Minimize the locality of the memory
	     `(min (+ ,@objects))))
      (let ((solution
	      (solve-problem
	       (parse-linear-problem
		objective
		`(,@constraints-on-c
		  ,@constraints-on-c1
		  ,@schedule-applied-indices
		  ,@iterator-constraints))))
	    (scale-solved (make-array `(,(length indices) ,(length indices)) :initial-element 0 :element-type 'fixnum)))
	(dotimes (i (length indices))
	  (dotimes (j (length indices))
	    (setf (aref scale-solved i j) (solution-variable solution (aref scale-to-minimize i j)))))
	;; (print scale-solved)
	;; (format t "~%Indices: ~a~%" indices)
	(let* ((solved-order
		 (loop for column-nth upfrom 0 below (length indices)
		       for list = (map 'list #'(lambda (x) (aref scale-solved column-nth x)) (range-list 0 (length indices)))
		       collect
		       (nth (position 1 list) indices)))
	       (solved-order
		 (remove-duplicates `(,@indices ,@(remove-duplicates solved-order)))))
	  ;; (format t "Solved Order: ~a~%" solved-order)
	  (when (not (= (length (remove-duplicates solved-order))
			(length (remove-duplicates indices))))
	    (return-from polyhedral-optimize schedule))
	  ;; (format t "Solved Order: ~a~%" solved-order)
	  ;; (print "Before Polyhedral")
	  ;; (print schedule)
	  ;; (print "After Polyhedral")
	  ;; (print (schedule-reorder schedule solved-order))

	  (let ((new-schedule (schedule-reorder schedule solved-order)))
	    (schedule-parallelize! new-schedule 0 n-threads)
	    new-schedule))))))

(defun schedule-collapse! (schedule)
  (declare (type Scheduler schedule))
  ;;
  ;; (C0)
  ;; (C1) @ (i j k) == (i j k)
  ;; (C2)
  ;;
  ;; I think it is beyond me ¯\_(ツ)_/¯
  ;; Let's think in the next PR
  schedule)

;; Usage Example
;; Einsum (matmul)
#+(or)
(let* ((out
 	 (make-indexspace
	  (wf/t:make-input `(1024 1024) :Z)
	  :subscripts `(i k)
	  :sizes (list 1024 1024)))
       (actions
	 (list
	  ;; ij jk ik
	  ;; (10 20) (20 10) (10 10)
	  (make-invocation
	   :einsum
	   (list
	    (make-indexspace
	     (wf/t:make-input `(1024 1024) :X)
	     :subscripts `(i j)
	     :sizes (list 1024 1024))
	    (make-indexspace
	     (wf/t:make-input `(1024 1024) :Y)
	     :subscripts `(j k)
	     :sizes (list 1024 1024))
	    out)
	   (list out)))))
  (time (print (solve-invocations actions))))

;;for (n in 0..batch)
;;for (fout in 0..out_features)
;;for (y in 1..H-1)
;;for (x in 1..W-1)
;;for (fin in 0..in_features)
;;for (k0 in 0..3)
;;for (k1 in 0..3)
;;conv[n, fout, y, x] += weigths[fout, fin, y, x] * input[n, fin, y+k0, x+k1];

;; Conv2D
;; TODO: x + y indexing rule...
#+(or)
(let* ((batch 10)
       (img-x 128)
       (img-y 128)
       (in-features 32)
       (out-features 32)
       (k-h 25)
       (k-w 25)
       (target
 	 (make-indexspace
	  (wf/t:make-input `(,batch ,out-features ,img-y ,img-x) :OUT)
	  :subscripts `(batch out-features img-y img-x)
	  :sizes `(,batch ,out-features ,img-y ,img-x)))
       (actions
	 (list
	  (make-invocation
	   :einsum ;; += *
	   (list
	    (make-indexspace
	     (wf/t:make-input `(,out-features ,in-features ,img-y ,img-x) :W)
	     :subscripts `(out-features in-features img-y img-x)
	     :sizes `(,out-features ,in-features ,img-y ,img-x))
	    (make-indexspace
	     (wf/t:make-input `(,batch ,in-features ,k-h ,img-y) :IN)
	     :subscripts `(batch in-features img-y img-x)
	     :sizes `(,batch ,in-features ,k-h ,k-w)
	     ;;:additional-offsets `(nil nil k0 k1)
	     )
	    target)
	   (list target))))
       (schedule (create-schedule-helper
		  `((batch . ,batch)
		    (out-features . ,out-features)
		    (img-y . ,(1- img-y))
		    (img-x . ,(1- img-x))
		    (in-features . ,in-features)
		    (k-w . ,k-h)
		    (k-h . ,k-w))
		  actions)))
  (print schedule)
  ;; SIMDifiy
  ;;(print (polyhedral-optimize schedule))
		  
  ;(time (solve-invocations actions))
  )


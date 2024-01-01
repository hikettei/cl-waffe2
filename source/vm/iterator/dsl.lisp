
(in-package :cl-waffe2/vm.iterator)

;;
;; A DSL to represent the iteration
;; Rules:
;;  - Unlike VM mode, iterations aren't collapsed
;;  - The operation is represented as:
;;    IndexSpace(x, [a*stride + offset, b*stride + offset, c*stride + offset, ...])
;;    =
;;    IndexSpace(x, ...)
;;  - A form is represented as: I_1 = op(I_2)

;;
;; TODO: Lisp-Like DSL representing an iterator
;;       Implementing gemm
;;       Implementing an parallelized and optimized einsum
;; APIs are like:
;;
;; for(i=10)  |
;;  for(k=10) | <- (make-iterators :x 1 :y 2 :z 3)
;;    {op}    |
;;
;; (! x a b c d) (make-reference)
;; (! y a b c d) (make-reference)
;;
;; (call (Lazy #'!sin :x 10 :y 20 :z 30)
;;       (! x :x :y :z)
;;       (! y :z :x :y))
;;
;; (%einsum) an alias for it.
;;


(defstruct Action
  "Action: IndexSpace(target) op IndexSpace(source)"
  (rank   0    :type fixnum)
  (depends nil :type list)
  (source nil  :type list)
  (target nil  :type list)
  (op))

(defstruct (IndexSpace
	    (:conc-name ispace-)
	    (:constructor
		%make-indexspace
		(tensor space)))
  (tensor)
  (space nil :type list))

(defstruct (IndexRef
	    (:conc-name iref-))
  (index nil :type (or keyword symbol))
  (size   0 :type fixnum)
  (offset 0 :type fixnum)
  (stride 0 :type fixnum))

(defmethod print-object ((act action) stream)
  (format stream "Action ~a: ~a <- ~a"
	  (action-op act)
	  (action-target act)
	  (action-source act)))

(defmethod print-object ((iref indexref) stream)
  (format stream
	  "[~a: ~a~a]"
	  (if (= (iref-stride iref) 0)
	      "R"  ;; = Reduce
	      "I") ;; = Index
	  (if (= (iref-stride iref) 0)
	      ""
	      (format
	       nil
	       "~a{=0..~a}*~a"
	       (iref-index iref)
	       (iref-size  iref)
	       (iref-stride iref)))
	  (if (= (iref-offset iref) 0)
	      ""
	      (format nil "+~a" (iref-offset iref)))))

(defun print-iref (iref stream)
  (format stream
	  "~a~a"
	  (if (= (iref-stride iref) 0)
	      "0"
	      (format
	       nil
	       "~a*~a"
	       (iref-index iref)
	       (iref-stride iref)))
	  (if (= (iref-offset iref) 0)
	      ""
	      (format nil "+~a" (iref-offset iref)))))

(defmethod print-object ((act IndexSpace) stream)
  (format stream "~a[~a]"
	  (tensor-id (ispace-tensor act))
	  (ispace-space act)))

(defun find-depends (&rest ispaces)
  (let ((found))
    (dolist (ispace ispaces)
      (dolist (ref (ispace-space ispace))
	(when (and
	       (not (= 0 (iref-stride ref)))
	       (not (= 1 (iref-size   ref))))
	  (push (iref-index ref) found))))
    (delete-duplicates found)))

(defun make-indexspace (tensor &key subscripts sizes)
  "Assertion: Symbols are determined"
  (when (some #'listp (tensor-stride tensor))
    (setf
     (tensor-stride tensor)
     (wf/t::calc-strides
      (wf/t::translate-adjustable-shape
       (wf/t::original-shape tensor))
      (order tensor))
     (tensor-stride tensor)
     (wf/t::sync (tensor-stride tensor) (reverse (wf/t::tensor-permute-order tensor)))))
  
  (let* ((n-dim (dims tensor))
	 (ranges
	   (loop for rank upfrom 0 below n-dim
		 for index in subscripts
		 for view in (tensor-view tensor)
		 for stride in (tensor-stride tensor)
		 for size   in sizes
		 collect
		 ;; List (size, offset, stride)
		 (let ((range (wf/t::subscript-range view)))
		   (make-indexref
		    :index index
		    :size  size
		    :offset (range-nth range 0)
		    :stride
		    (if (wf/t::subscript-broadcast view)
			0
			(*
			 (- (range-nth range 1) (range-nth range 0))
			 stride)))))))
    (%make-indexspace tensor ranges)))

;; (defun %einsum-helper)

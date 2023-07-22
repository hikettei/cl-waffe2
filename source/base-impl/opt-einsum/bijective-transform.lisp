
(in-package :cl-waffe2/base-impl)

;; Here, we provide an implementation of einsum dedicated to form like: A -> B, from an one tensor to an one tensor.

;; No GPU Support?
#|
(defnode (Bijective-Einsum-Transform (self iterators from to)
	  :slots ((iterators :initarg :iterators :initform nil :reader read-iterators))
	  :where (A[from] out[to] -> out[to])))

(define-impl (Bijective-Einsum-Transform :device t)
	     :forward ((self x out)
		       `(progn
			  (cl-waffe2:with-facets ((x* (,x :direction 'array :sync t))
						  (out* (,out :direction 'array :sync t)))
			    ,@(expand-dolist-form out* x* (read-iterators self)))
			  ,x))
	     :backward ((self dout x out)

			))

(defun expand-dolist-form (vec-out vec1 iterators)
  (labels ((expand-dolist (rest-iterators)
	     (if rest-iterators
		 `(dolist (,(car rest-iterators) 
  )
;; Make it contiguous in advance

;; with-facetで色々やる


(defun make-transform (from-me-to-me ;; A[i j] -> A[j i]
		       before-symbol
		       after-symbol
		       
		       from-shape
		       out-shape
		       declared
		       &key
			 (broadcast '~))
  
  ;; If after-symbol  = nil
  ;; If from-me-to-me = t, do it in-place.
  
  ;; S-exp is returned

  (let ((iterators (determine-iterator-shapes from-shape out-shape declared broadcast)))

    (print iterators)
    nil
))
|#

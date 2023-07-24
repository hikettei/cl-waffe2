
(in-package :cl-waffe2/vm.nodes)

;;
;; Forward [Shape-Error]
;;  |
;; [apply-broadcast] (If there's flexible axes)
;;  |
;; Forward (If failed, never broadcasted)
;;  |
;; out

;;
;; (3 -1 3)
;; (3 3 -1)
;;
;; (-1 3 3)
;;    (3 3)
;;

(defun inject-1 (tensor uprank-n)
  (alexandria:flatten
   (loop for s in (shape-with-broadcastable tensor)
	 if (equal s -1)
	   collect (make-list uprank-n :initial-element -1)
	 else
	   collect s)))

(defun apply-broadcast (input-states inputs uprankable-list)
  (let* ((largest-axis (loop for i in input-states
			     for tensor in inputs
			     unless (tensor-flexible-p tensor)
			       maximize (length i)))
	 (uprank-from-flexible-p (= largest-axis 0))
	 (largest-axis
	   (if uprank-from-flexible-p ;; every inputs are flexible
	       (loop for i in input-states
		     maximize (length i)) ;; including -1 axis.
	       largest-axis))
	 (reference-list
	   (loop for input in inputs
		 for uprankable in uprankable-list
		 collect (inject-1 input (- largest-axis (length (shape input)))))))
    ;; The :where is...
    ;; [~ x y] <- it is ok to apply uprank rule.
    ;; [x y]   <- it is ng to apply uprank rule.


    (loop for input in inputs
	  for uprankable in uprankable-list
	  for references in reference-list
	  if (and (tensor-flexible-p input) ;; (3 -1 -1 -1 3)
		  uprankable)
	    collect (let* ((rankup-n  (- largest-axis (length (shape input))))
			   (out (cl-waffe2/base-impl:!rankup
				 input
				 rankup-n
				 :at (tensor-flexible-p input)))
			   (view-args (loop for ref in references
					    for dim upfrom 0
					    if (equal ref -1)
					      collect
					    `(:broadcast ,(nth dim (find -1 reference-list :key #'(lambda (list) (nth dim list)) :test #'(lambda (x y) (not (shape-equal x y))))))
					    else
					      collect t))
			   (out (apply #'cl-waffe2/base-impl:!view out view-args)))
		      ;; Apply Broadcast to flexible axis
		      (setf (tensor-flexible-p out) nil)
		      out)
	  else
	    collect input)))



(in-package :cl-waffe2/vm.nodes)

;;
;; Forward [Shape-Error]
;;  |
;; [apply-broadcast] (If there's flexible axes)
;;  |
;; Forward (If failed, never broadcasted)
;;  |
 ;; out
(defun apply-broadcast (input-states inputs uprankable-list)
  (let* ((largest-axis (loop for i in input-states
			     for tensor in inputs
			     unless (tensor-flexible-p tensor)
			       maximize (length i)))
	 (largest-axis
	   (if (= largest-axis 0) ;; every inputs are flexible
	       (loop for i in input-states
		     maximize (length i))
	       largest-axis))
	 (largest-axis-shape
	   (shape
	    (find largest-axis inputs
		  :test #'=
		  :key #'(lambda (x) (length (shape x)))))))
    ;; The :where is...
    ;; [~ x y] <- it is ok to apply uprank rule.
    ;; [x y]   <- it is ng to apply uprank rule.

    (loop for input in inputs
	  for uprankable in uprankable-list
	  if (and (tensor-flexible-p input)
		  uprankable)
	    collect (let* ((rankup-n (- largest-axis (length (shape input))))
			   (out (cl-waffe2/base-impl:!rankup
				 input
				 rankup-n
				 :at (tensor-flexible-p input)))
			   (view-pad (loop for i upfrom 0 below (tensor-flexible-p input)
					   collect t))
			   (subscripts (loop with bias = (tensor-flexible-p input)
					     for i upfrom 0 below rankup-n
					     collect `(:broadcast ,(nth (+ bias i) largest-axis-shape))))
			   (out (apply #'cl-waffe2/base-impl:!view out `(,@view-pad ,@subscripts))))
		      ;; Apply Broadcast to flexible axis
		      (setf (tensor-flexible-p out) nil)
		      out)
	  else
	    collect input)))


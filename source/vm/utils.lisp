
(in-package :cl-waffe2/vm)

(defun compose (&rest fns)
  (if fns
      (let ((fn1 (car (last fns)))
            (fns (butlast fns)))
        #'(lambda (&rest args)
                   (reduce #'funcall fns
                           :from-end t
                           :initial-value (apply fn1 args))))
      #'identity))

;; Ref: http://www.utkuevci.com/ml/autograd/
(defun topological-sort (var)
  (declare (type AbstractTensor var))
  (let ((seen nil)
	(top-sort nil))
    (declare (type list seen top-sort))
    (labels ((top-sort-helper (v is-leaf-p)
	       (if (or (find (tensor-iid v) seen :key #'tensor-iid :test #'eql)
		       ;;(null (tensor-backward v))
		       is-leaf-p)
		   nil
		   (progn
		     (push v seen)
		     (dolist (prev (tensor-variables v))
		       (top-sort-helper prev (detach-p v)))
		     (push v top-sort)))))
      (top-sort-helper var (detach-p var))
      (reverse top-sort))))


;; sort-and-prune-for-backward:
;;     tp-sorted   => Pruned
;;        X    x
;;      copy(x)|
;;          \ /             x
;;     X    sin             |
;;   copy(x) |     =>      sin
;;      \   /               |
;;       sin               sin
;;        |                 |
;;       out               out

(defun sort-and-prune-for-backward (toplevel dout-toplevel leaves)
  (declare (type AbstractTensor toplevel))
  (let ((seen nil))
    (labels ((top-sort-helper (var prev-gradient)
	       (let ((encounter-x (find (tensor-iid var) seen :test #'eql))
		     (found-param  (or (null (tensor-backward var))
				       (null (tensor-variables var)))))
		 (if (or encounter-x found-param)
		     (cond
		       (encounter-x nil)
		       (found-param
			(when (slot-value var 'cl-waffe2/vm.generic-tensor::requires-grad)
			  ;; Gradient Adder wo tukuru
			  `(,(expand-gradient-adder var prev-gradient)))))
		     (let ((bw (apply
				#'cl-waffe2/vm.nodes:compiler-expand-backward
				(tensor-backward var)
				prev-gradient
				(tensor-variables var)))
			   (above-sort nil))
		       (push (tensor-iid var) seen)
		       (loop for prev in (tensor-variables var)
			     for grad in bw
			     for nth fixnum upfrom 0
			     if grad do
			       (let* ((result (top-sort-helper prev grad)))
				 (when result
				   (multiple-value-bind (bwnode iseq-printer) (make-backward-instruction var prev-gradient nth leaves)
				     (setq above-sort
					   `(,@above-sort
					     ,(make-wfop
					       bwnode
					       grad
					       iseq-printer
					       (list prev-gradient))			
					     ,@result))))))
		       above-sort)))))
      (top-sort-helper toplevel dout-toplevel))))



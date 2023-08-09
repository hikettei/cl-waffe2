
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
      (top-sort-helper var nil)
      (reverse top-sort))))

;; TODO: Delete Save4bw
(defun topological-sort-in-backward-direction (var dout-toplevel)
  (declare (type AbstractTensor var dout-toplevel))
  (let ((seen nil)
	(top-sort nil))
    (declare (type list seen top-sort))
    (labels ((top-sort-helper (v prev-dout)
	       (if (or (find (tensor-iid v) seen :key #'tensor-iid :test #'eql)		       
		       (null (tensor-backward v))
		       (null prev-dout))
		   nil
		   (let ((bw-directions		  
			   (apply
			    #'cl-waffe2/vm.nodes:compiler-expand-backward
			    (tensor-backward v)
			    prev-dout
			    (tensor-variables v)))
			 (prev-vars (tensor-variables v)))
		     (push v seen)
		     (loop for prev in (reverse prev-vars)
			   for grad in (reverse bw-directions)
			   if (and prev (ancestor-param-p prev))
			     do (top-sort-helper prev grad))
		     (when (ancestor-param-p v)
		       (push `(,prev-dout ,v) top-sort))))))
      (top-sort-helper var dout-toplevel)
      (reverse top-sort))))


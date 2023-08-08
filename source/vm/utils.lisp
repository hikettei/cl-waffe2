
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


(defun topological-sort-iseq (iseq)
  (declare (type list iseq))
  (let ((seen nil)
	(top-sort nil)
	(grad-adders nil))
    (declare (type list seen top-sort))
    (labels ((top-sort-helper (i)
	       (if (or (find (tensor-iid (wfop-self i)) seen :key (compose #'tensor-iid #'wfop-self) :test #'eql)
		       (wfop-bw-is-leaf-p i))
		   (when (wfop-bw-is-leaf-p i)
		     (push i grad-adders))
		   (progn
		     (push i seen)
		     (push i top-sort)))))
      (mapc #'top-sort-helper iseq)
      (values
       (reverse top-sort)
       (remove-duplicates
	grad-adders
	:test #'equal
	:key #'(lambda (x)
		 (append
		  (list (tensor-id (wfop-self x)))	 
		  (map 'list #'tensor-id (wfop-args x)))))))))


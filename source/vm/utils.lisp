
(in-package :cl-waffe2/vm)

;; Ref: http://www.utkuevci.com/ml/autograd/

(defun topological-sort (toplevel)
  (declare (type AbstractTensor toplevel))

  )

(defun topological-sort (var)
  (let ((seen nil)
	(top-sort nil))
    (labels ((top-sort-helper (v is-leaf-p)
	       (if (or (find (tensor-id v) seen :key #'tensor-id :test #'eql)
		       (null (tensor-backward v))
		       is-leaf-p)
		   nil
		   (progn
		     (push v seen)
		     (dolist (prev (tensor-variables v))
		       (top-sort-helper prev (detach-p v)))
		     (push v top-sort)))))
      (top-sort-helper var nil)
      top-sort)))


(in-package :cl-waffe2/vm.nodes)

;; I'm depcrecated with Composite-Function
;; Should be deleted in the future release:


;; The file function.lisp provides a system on interacting Lazy-Evaluated Nodes and Toplevel functions:
;;
;; Composite <-> Function
;; Node      <-> Function
;;

;; One composite -> a single defun form.
;; In order to implemenet "GENERIC" behaviour, we need to wrap composite->defun by higher-order function.

(defun eliminate-undetermined-size (tensor)
  (let* ((shape (cl-waffe2/vm.generic-tensor::translate-adjustable-shape (actual-shape tensor)))
	 (out (make-input  shape nil
			   :create-from tensor
			   :dtype (dtype tensor)
			   :order (order tensor)
			   :scalar-p (scalar-p tensor)))
	 (broadcasted-p)
	 (broadcasts (loop for size in (cl-waffe2/vm.generic-tensor::translate-adjustable-shape (shape tensor))
			   for view in (tensor-view tensor)
			   if (eql :broadcast (cl-waffe2/vm.generic-tensor::viewtype (cl-waffe2/vm.generic-tensor::force-list view)))
			     collect (and
				      (setq broadcasted-p t)
				      `(:broadcast ,size))
			   else
			     collect t))
	 (out (if broadcasted-p
		  (apply #'cl-waffe2/vm.generic-tensor::view out broadcasts)
		  out)))
    
    (setf (cl-waffe2/vm.generic-tensor::tensor-initial-offset out) (cl-waffe2/vm.generic-tensor::tensor-initial-offset tensor)
	  (tensor-vec out) (cl-waffe2/vm.generic-tensor::vec tensor))
    out))


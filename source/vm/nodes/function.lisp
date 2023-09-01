
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
  (let ((place (make-input
		(cl-waffe2/vm.generic-tensor::translate-adjustable-shape (shape tensor))
		nil
		:create-from tensor
		:dtype (dtype tensor)
		:order (order tensor)
		:scalar-p (scalar-p tensor))))
    (cl-waffe2/vm.generic-tensor::embody-tensor-vec place tensor)
    ;; set facet = :exist?
    place))


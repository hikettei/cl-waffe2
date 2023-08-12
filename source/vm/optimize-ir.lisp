
(in-package :cl-waffe2/vm)

;; TODO: Add PermuteNode
(defparameter *special-position-nodes* '(cl-waffe2/base-impl::ViewTensorNode cl-waffe2/base-impl::ReshapeNode)) ;; + ReshapeNode PermuteNode

(defun shuffle-node-p (node)
  (some #'(lambda (n)
	    (subtypep (class-of node) n))
	*special-position-nodes*))

;; Optimization to cl-waffe2 IR:
;; = 1. View Reordering ======

;; VIEW do not necessary need to follow the position in iseq. should be placed anywhere.

;; Example
;; ScalarMUL    VIEW    }
;; VIEW         VIEW    } collecting View/Reshape/Permute Nodes and placing them to the top of iseq
;; ADD       => ScalarMul
;; VIEW         ADD
;; To Compose More Operations

(defun apply-iseq-reordering (iseq)
  "Shuffles the position of ViewTensorNode/ReshapeNode/PermuteNode to compose more operations."
  (declare (type list iseq))

  (let ((top-area)
	(rest-area)
	(ends-with ;; If the iseq ends with View, Copying View to the corresponding position to get the returned result
	  (when (shuffle-node-p (wfop-node (car (reverse iseq))))
	    (car (reverse iseq)))))
		       

    (loop for instruction in iseq
	  if (shuffle-node-p (wfop-node instruction))
	    do (push instruction top-area)
	  else
	    do (push instruction rest-area))
    `(,@(reverse top-area)
      ,@(reverse rest-area)
      ,ends-with)))

(defun apply-fuse-operations (iseq)
  (declare (type list iseq))
  iseq)


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

;; Currently this feature isn't used anymore!
(defun apply-iseq-reordering (iseq)
  "Shuffles the position of ViewTensorNode/ReshapeNode/PermuteNode to compose more operations."
  (declare (type list iseq))
  
  (return-from apply-iseq-reordering iseq)

  ;; [FixME] Disabled for a while because it's unstable!!!!!
  ;; And sometime produces an unexcepted behaviour!!!

  (when (null iseq) (return-from apply-iseq-reordering))

  (let ((top-area)
	(rest-area)
	(ends-with ;; If the iseq ends with View, Copying View to the corresponding position to get the returned result
	  (when (shuffle-node-p (wfop-node (car (reverse iseq))))
	    (list (car (reverse iseq))))))
		       

    (loop for instruction in iseq
	  if (shuffle-node-p (wfop-node instruction))
	    do (push instruction top-area)
	  else
	    do (push instruction rest-area))

    `(,@(reverse top-area)
      ,@(reverse rest-area)
      ,@ends-with)))

(defun apply-fuse-operations (iseq leaves)
  (declare (type list iseq leaves))

  ;; iseq:
  ;; out = A * B + C
  ;; 0 MUL
  ;; 1 ADD
  ;; 2 ...
  ;;
  ;; If 0 and 1 is composable, replaces 0 and 1 with (0 . 1).

  (apply-in-place-mutation! iseq leaves)
  
  (let ((result))
    (loop for inst of-type WFInstruction in iseq
	  do (let* ((last-val (car result))
		    (last-iseq-iter (when last-val
				      (or
				       (wfop-call-with-view last-val)
				       (tensor-iter-of (wfop-self last-val)))))
		    (current-op-iter (tensor-iter-of (wfop-self inst)))
		    (compose-p (and
				(it.-able-p last-iseq-iter current-op-iter)
				(if (movetensor-p (wfop-node inst))
				    (progn
				     ;; <Deleted> Node isn't subject to FuseOps
				     ;; Because the costs for it is almost 0
				     ;; and which tensors to be returned is still unknown
				     (not (movetensor-ignore-me (wfop-node inst))))
				    t)
				(if (and last-val
					 (movetensor-p (wfop-node last-val)))
				    (not (movetensor-ignore-me (wfop-node last-val)))
				    t))))
	       
	       (if compose-p
		   (let ((latest (pop result)))
		     ;;(print "COMPOSE!")
		     ;;(print latest)
		     ;;(print inst)
		     (push (compose-two-ops latest inst) result))
		   (progn
		     (push inst result)))))
    (accept-all-lazy-compile! result)
    ;; [TODO] Shuffling to get more fusions, esp: ScalarMulNode in !sum
    (reverse result)))

(defun accept-all-lazy-compile! (iseq)
  (loop for inst of-type WfInstruction in iseq
	if (wfop-fused-body-cache inst)
	  do (setf (wfop-op inst)
		   (compile nil
			    (make-callable-fused-f
			     (wfop-call-with-view   inst)
			     (wfop-fused-body-cache inst)
			     (tensor-id (wfop-self  inst)))))))


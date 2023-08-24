
(in-package :cl-waffe2/vm)

;; This file isn't used anymore for a while
;; TODO:
;;  Add a macro named `defpath`
;;  We create a search-based fuse-ops, which is user-extensible

;; defpath possess these datum:
;;  1. Nodes to be replaced
;;  2. After replaced IR

;; For example, (!add (!mul 2.0 x) y) should be expressed as (axpy! 2.0 x 0.0 y) to reduce instructions.
;; defpath is the macro to achieve this ^

;; ~~~ Developing Cycle with cl-waffe2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  1. Declare the new device (e.g.: CPUTensor)
;;  2. Prepare allocator and accessors (e.g.: initialize-instance method, vref and (setf vref))
;;  3. Implement existing operations with define-impl
;;  4. Blush up the generated IR with defpath macro to fuse more operations in a small cycle.
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defparameter *user-defined-path-list* (make-hash-table))

(defun reset-all-path! ()
  (setf *user-defined-path-list* nil))

(defstruct (FusionPathQuerySet
	    (:conc-name qset-)
	    (:constructor make-query-set
		(name reject-p query-list replace-form)))
  (name name :type Symbol)
  (reject-p reject-p :type function)
  (query-list query-list :type list)
  (replace-form replace-form :type function))

(defstruct (FusionPathQuery
	    (:conc-name query-)
	    (:constructor make-query
		(abstract-node
		 &key
		   (device t)
		   (dtype  t)
		   (pred   #'(lambda (node) (declare (ignore node)) t)))))
  "
## [struct] FusionPathQuery

`(make-query ...)` will create a new query

`FusionPathQuery` becomes t when satisfies all of following conditions:

`abstract-node[symbol]` become t when the node is a subtype of `abstract-node`

`device[t or symbol]`   become t when the node is working under the device

`dtype[t or list]`      become t when the `dtype` is set to t, or the list of dtype in arguments are corresponds with the list. (e.g.: `(list :float :float)`)

`pred[function]`        specifies an additional predicator, the function receives `(node)` as arguments and return t to accept it. (`arguments-tensor` is an list of tensors, which `forward` or `call` used.)
"
  (node   abstract-node :type symbol)
  (device device :type symbol)
  (dtype  dtype  :type (or symbol keyword list))
  (pred   pred   :type function))

(defmacro defpath ((fusion-name &rest query-list) &key (reject-p nil) (replaced-with nil))
  "
## [macro] defpath

```lisp
(defpath (fusion-name &rest query-list) &key (reject-p #'(lambda ())) (replaced-with nil))
```

```lisp
Implementing cl-waffe2 to new devices.

 1. Declare the new device (e.g.: CPUTensor)
 2. Prepare allocator and accessors (e.g.: initialize-instance method, vref and (setf vref))
 3. Implement existing operations with define-impl
 4. Blush up the generated IR with defpath macro to fuse more operations in a small cycle. <- defpath, here!
```

The created and registered path, will be reset with the `(reset-all-path!)` function. All registered paths are visible in `*user-defined-path-list*` parameter.

## Rules

cl-waffe2 replaced the existing operations with following the rules below:

1. The search is performed ignoring SaveForBackwardNode. If it is contained in the area to be replaced, it is performed after the replacement.

```
Rule: [A] [B] -> [C]

{Before fused}

[A]
[SAVE_FOR_BACKWARD]
[B]

{After fused}

[C]
[SAVE_FOR_BACKWARD]
```

(TODO: Docstring)
"
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (setf (gethash ',fusion-name *user-defined-path-list*)
	   (make-query-set ',fusion-name
			   (or ,reject-p #'(lambda ()))
			   (list ,@query-list)
			   (lambda ,(car replaced-with) (progn ,@(cdr replaced-with)))))))


#|
(defpath (CPUReLUFusion-No-Grad
	  (make-query 'Where-Operation-Node :device 'cl-waffe2/backends.cpu:CPUTensor :dtype t)
	  (make-query 'MulNode              :device 'cl-waffe2/backends.cpu:CPUTensor :dtype t) ;; AbstractElwiseOperation | AbstractComparisonOperation | AbstractMathematicalOperation
	  )
	 ;;:reject-p #'simd-extension-p
	 :replaced-with ((&rest nodes)
			 (print nodes)
			 (!move
			  (wfop-self (car (last nodes)))
			  (!sin (cl-waffe2/distributions:randn `(3 3))))
))
|#
#|
(defpath (CPUReLUFusion-Diff
	  (make-query 'WhereOperationNode :device 'CPUTensor :dtype t)
	  (make-query 'MoveTensorNode     :device 'CPUTensor :dtype t)
	  (make-query 'MulNode            :device 'CPUTensor :dtype t))
	 :replaced-with ((where move mul)

))
|#


(defun query-match-p (query inst)
  (declare (type FusionPathQuery query)
	   (type WfInstruction inst))
  (let ((node (wfop-node inst)))
    (and
     (subtypep (class-of node) (find-class (query-node query)))
     (subtypep (class-of (wfop-self inst)) (find-class (query-device query)))
     (or (eql (query-dtype query) t)
	 (if (listp (query-dtype query))
	     (every #'(lambda (x y)
			(eql x (dtype y)))
		    (query-dtype query)
		    (wfop-args inst))
	     (eql (query-dtype query) (dtype (wfop-self inst)))))
     (funcall (query-pred query) (wfop-self inst)))))

;; Test this:
(defun apply-path-fusion (iseq &key (limit 3) (count 0))
  "`apply-path-fusions` start searching all replaceable combination of InstructionSeq declared via `defpath`, and replaces the IR.
The operation will continue until count=limit or there's no changes."
  (declare (type list iseq))

  ;; Count exceeds limit
  (when (>= count limit)
    (return-from apply-path-fusion iseq))
  
  (let ((no-changed-p t))
    (flet ((apply-fuse-helper (iseq-op query-set &aux (iseq-list nil))
	     (loop with query-count fixnum = 0
		   with query-list  list   = (reverse (qset-query-list query-set))
		   with sv4bw-stack list   = nil
		   with candidates  list   = nil
 	           for inst of-type WfInstruction in iseq-op
		   if (or (not (movetensor-p (wfop-self inst)))
			  (not (cl-waffe2/base-impl:mv-lazy-sv4bw (wfop-node inst))))
		     do (let ((match-p (query-match-p (nth query-count query-list) inst)))
			  ;; If matched, incf the counter, otherwise, set to 0
			  (if match-p
			      (progn
				(incf query-count 1)
				(if (null (nth query-count query-list))
				    ;; If reached to the last
				    (progn
				      (push inst candidates)
				      ;; Replace with ...

				      ;; [TODO] detach
				      ;; [TODO] pass WfInstruction
				      (dolist (instruction
					       (reverse
						(node-compile-into-vm
						 (apply
						  (qset-replace-form query-set)
						  candidates)
						 :fuse-p nil)))
					(push instruction iseq-list))
				      (dolist (mv sv4bw-stack)
					(push mv iseq-list))
				      (setq candidates nil)
				      (setq sv4bw-stack nil)
				      (setq no-changed-p nil)
				      (setq query-count 0))
				    (progn
				      ;; Stack to the candidates
				      (push inst candidates))))
			      (progn
				(dolist (mv sv4bw-stack)
				  (push mv iseq-list))
				(setq sv4bw-stack nil)
				(setq query-count 0)
				(push inst iseq-list))))
		   else
		     do (progn
			  ;; Remains (MoveTensorNode(SaveForBackward))
			  (push inst sv4bw-stack))
		   finally (progn
			     (dolist (c candidates)
			       (push c iseq-list))
			     (dolist (mv sv4bw-stack)
			       (push mv iseq-list))))
	     iseq-list))

      (maphash
       #'(lambda (name op)
	   (declare (ignore name))
	   (setq iseq (apply-fuse-helper (reverse iseq) op)))
       *user-defined-path-list*)
      
      ;; apply-path-fusion is forcibly called recursively until there's no modification
      (if no-changed-p
	  iseq
	  (apply-path-fusion iseq :limit limit :count (1+ count))))))


;;
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; The code below is an attempt to elwise operation fusion
;; But turned out to be not working well!
;; And the equivalent feature should be implemented by defpath macro!
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; TODO: These things are subject to be deleted in the future release!
;;
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
				(not (broadcasted-p (wfop-self inst)))
				(it.-able-p last-iseq-iter current-op-iter)
				(no-dependency-p last-val inst)
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
			     inst)))))


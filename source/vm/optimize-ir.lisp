
(in-package :cl-waffe2/vm)

;; This file isn't used anymore for a while
;; TODO:
;;  Add a macro named `defpath`
;;  We create a search-based fuse-ops, which is user-extensible

;; defpath possess these datum:
;;  1. Nodes to be replaced
;;  2. After replaced IR

;; For example, (!add (!mul 2.0 x) y) should be expressed as (axpy! 2.0 x 0.0 y) to reduce instructions.
;; defpath is the macro to achieve this ^ (i.e.: Partially applying Symbolic Differentiation like Theano autodiff)

;; ~~~ Developing Cycle with cl-waffe2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  1. Declare the new device (e.g.: CPUTensor)
;;  2. Prepare allocator and accessors (e.g.: initialize-instance method, vref and (setf vref))
;;  3. Implement existing operations with define-impl
;;  4. Blush up the generated IR with defpath macro to fuse more operations in a small cycle.
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defparameter *user-defined-path-list* (make-hash-table))

(defun reset-all-path! ()
  "
## [function] reset-all-path!

`(setf *user-defined-path-list* (make-hash-table))`"
  (setf *user-defined-path-list* (make-hash-table)))

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

`(make-query ...)` and create a new query.

A single `FusionPathQuery` becomes t only when satisfies all of following conditions:

`abstract-node[symbol]` become t when the node is a subtype of `abstract-node`

`device[t or symbol]`   become t when the node is working under the device or `subtype` of it.

`dtype[t or list]`      become t when the `dtype` is set to t, or the list of dtype in arguments are corresponds with the list. (e.g.: `(list :float :float)`)

`pred[function]`        specifies an additional predicator, the function receives `(node)` as arguments and return t to accept it. (`arguments-tensor` is an list of tensors, which `forward` or `call` used.)

See also: `defpath`.
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

Define a `FusionQueryPath` to relocate compiled instructions with reference to the search. Composing the sequence of generated IRs to suit the device or model is the easiest way to speed up your model, cl-waffe2 searches for compiled nodes and replaces those matching the conditions specified in `query-list` with the computed nodes specified in `replaced-with`, if `:fuse-p` is set to t (default: `t`). In the simplest case, `defpath` can detect `[AddNode-CPUTensor] [MulNode-CPUTensor]` sequence and replace it with `[AddMulNode-CPUTensor]` node to reduce the number of instructions.

```lisp
[When adding a new device to cl-waffe2...]
 1. Declare the new device (e.g.: CPUTensor, CUDATensor ...)
 2. Prepare allocator and accessors (e.g.: initialize-instance method, vref and (setf vref))
 3. Implement existing operations with define-impl macro
 4. Blush up the generated IR with defpath macro to fuse more operations in a small cycle. <- defpath, here!
```

The created and registered path, will be reset with the `(reset-all-path!)` function. All registered paths are stored in `*user-defined-path-list*` parameter.

## Rules

cl-waffe2 replaces the existing operations with following the rules:

1. The search is performed ignoring SaveForBackwardNode. If it is contained in the area to be replaced, it is moved to the last sequence of replaced one.


```
Example:

Rule: [A] [B] -> [C]
```

```
Before Fusion:

[A]
[SAVE_FOR_BACKWARD]
[B]
[M]
[N]
```

```
Searching will be done ignoring [SAVE_FOR_BACKWARD]

^ [A]
| [B]
| [M]
| [N]
reading in this direction.
```

```lisp
After fusion:

[C]  ;; [A] [B] -> [C]
[SAVE_FOR_BACKWARD] ;; placed after the operation
[M]
[N]
```

2. `defpath` priority is given to those registered first.

Repeat the search until no more targets are found to replace it.

3. query-list

Not replaced until the `query-list` matches everything, including the order.

### Example

(TODO: For the case of ReLU)

### make-query

### WfInstruction

"
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (setf (gethash ',fusion-name *user-defined-path-list*)
	   (make-query-set ',fusion-name
			   (or ,reject-p #'(lambda ()))
			   (list ,@query-list)
			   (lambda ,(car replaced-with) (progn ,@(cdr replaced-with)))))))

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

(defun detach-all-iseq! (iseq)
  (dolist (i iseq)
    (setf (detach-p (wfop-self i)) t)
    (dolist (arg (wfop-args i))
      (setf (detach-p arg) t))))

(defun undo-detach-all-iseq! (iseq)
  (dolist (i iseq)
    (setf (detach-p (wfop-self i)) nil)
    (dolist (arg (wfop-args i))
      (setf (detach-p arg) nil))))

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
				      (detach-all-iseq! candidates)
				      (dolist (instruction
					       (reverse
						(node-compile-into-vm
						 (apply
						  (qset-replace-form query-set)
						  candidates)
						 :fuse-p nil)))
					(push instruction iseq-list))
				      (undo-detach-all-iseq! candidates)
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



(in-package :cl-waffe2/vm)


(defstruct (WFInstruction
	    (:conc-name wfop-)
	    (:constructor make-wfop (op self node args &key (fuse-prev nil) (fused-body-cache nil) (call-with-view nil))))
  "
## [struct] WFInstruction

Instruction: Sets the result of λ function op called with `args`, into self.state.forward_result

Basically follows this format:

 out_target <- Args1 Args2 Args3 ...

cl-waffe2 vm specializes on  the sequence of above format.
"
  (op   op   :type function)
  (node node :type (or function string null AbstractNode))
  (self self :type AbstractTensor)
  (args args :type list)
  (bw-is-leaf-p nil :type boolean)
  (call-with-view call-with-view :type (or null cl-waffe2/vm.generic-tensor::Ranked-Loop))
  (fuse-prev fuse-prev :type (or null list))
  (fused-body-cache fused-body-cache :type (or null list)))

;; (defstruct (Composable-Operator <- separate call-with-view from body
;; (defun .cop (cop1 cop2) ...)

(defparameter *omit-args-n* 5)
(defparameter *opname-indent-to* 0 "Adds a space for this param times")

(defmethod print-object ((inst WFInstruction) stream)
  (format stream
	  "~a~a : ~a <= op(~a)>~%"
	  (instruction-opname inst)
	  (with-output-to-string (out)
	    (dotimes (i (- *opname-indent-to* (length (instruction-opname inst)))) (princ " " out)))
	  (tensor-id (wfop-self inst))
	  ;;(shape (wfop-self inst))
	  (if (>= (length (wfop-args inst)) *omit-args-n*)
	      (format nil "..., x~a,..." (length (wfop-args inst)))
	      (with-output-to-string (out)
		(dotimes (i (length (wfop-args inst)))
		  (let ((var (nth i (wfop-args inst))))
		    (format out "~a~a~a~a"
			    (if (slot-value var 'cl-waffe2/vm.generic-tensor::requires-grad)
				"<Param>"
				"")
			    (tensor-id var)
			    (shape var)
			    (if (nth (1+ i) (wfop-args inst))
				" "
				""))))))))

(defmethod instruction-opname ((inst WFInstruction))
  (format nil "<WfInst[Compiled: ~a]"
	  (if (functionp (wfop-node inst))
	      (funcall (wfop-node inst))
	      (if (movetensor-p (wfop-node inst))
		  (if (movetensor-ignore-me (wfop-node inst))
		      "<DELETED>"
		      (if (cl-waffe2/base-impl:mv-lazy-sv4bw (wfop-node inst))
			  (if (scalar-p (wfop-self inst))
			      "MoveScalarNode(SAVE_FOR_BACKWARD)"
			      "MoveTensorNode(SAVE_FOR_BACKWARD)")
			  (class-name (class-of (wfop-node inst)))))
		  (class-name (class-of (wfop-node inst)))))))

(defun area-indent-to (iseq)
  "Returns the largest length of iseq name"
  (loop for i in iseq
	if (not (functionp (wfop-node i)))
	  maximize (length (instruction-opname i))))

(defmacro with-indent-to (iseq &body body)
  `(let ((*opname-indent-to* (area-indent-to ,iseq)))
     ,@body))

;; In-place mutation

;;      v judge:is this usage of A is the last?
;; A <- A B
;;   ...
;; K <- A B

(defun apply-in-place-mutation! (iseq leaves)
  (declare (type list iseq leaves))
  
  (let ((ref-table (make-hash-table)))

    ;; First, Register all tensors appeared in the computation node
    (mapc
     #'(lambda (variable)
	 ;; Tensors that can be destructed is:
	 ;; InputTensor
	 ;; Set 100 as for ExistTensor, in order not to destruct training data/parameters

	 (if (eql (tensor-attribute variable) :chain)
	     ;;(setf (gethash (tensor-id variable) ref-table) 0) ;; [FixME] Originally it was set to 0, but here's unexcepted side effects...
	     (setf (gethash (tensor-id variable) ref-table)
		   (if *no-grad*
		       0
		       1))
	     (setf (gethash (tensor-id variable) ref-table) nil)))
     leaves)

    ;; Tracing all the computation nodes, counting up reference tables
    ;; X <- A B : A B ... reference, X ... set
    (mapc
     #'(lambda (instruction)
	 (when (not (movetensor-p (wfop-node instruction)))
	   (mapc
	    #'(lambda (arg)
		(when (gethash (tensor-id arg) ref-table)
		  (incf (gethash (tensor-id arg) ref-table) 1)))
	    (wfop-args instruction))))
     iseq)

    ;; Based on ref-table, we retribute whether MoveTensor should be ignored or not.

    (mapc
     #'(lambda (instruction)
	 (when (and (movetensor-p (wfop-node instruction))
		    (not (sv4bw-p (wfop-node instruction)))) ;; save4bw never ignored!
	   ;; MoveTensor: A B -> A (Place Target -> Place)
	   ;; In-place MoveTensor: A B -> B

	   (let* ((bw     (wfop-node instruction))
		  (past-variables (tensor-variables (wfop-self instruction)))
		  (target (second (wfop-args instruction)))
		  (in-place-p
		    (and
		     (gethash (tensor-id target) ref-table)
		     (<= (gethash (tensor-id target) ref-table) 1)

		     ;; :force t is not subject to in-place

		     ;; The problem is that: it is unknown wheter movetensor returns Viewed Input or not.
		     ;; So Tensors whose place has multi-dimensional offset, is never deleted.

		     (apply #'cl-waffe2/vm.generic-tensor::order-reductable-p 0 past-variables) ;; <- is it worth it? test
		     (not (tensor-protect-me (car past-variables)))
		     (not (movetensor-save-for-backward bw)))))	     
	     (if in-place-p
		 (setf (movetensor-ignore-me (wfop-node instruction)) t)
		 (when (gethash (tensor-id target) ref-table)
		   (decf (gethash (tensor-id target) ref-table)))))))
     iseq)
    nil))


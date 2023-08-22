
(in-package :cl-waffe2/vm)


(defstruct (WFInstruction
	    (:conc-name wfop-)
	    (:constructor make-wfop (op self node args &key (fuse-prev nil) (fused-body-cache nil) (call-with-view nil))))
  "
## [struct] WFInstruction

Instruction: Sets the result of λ function op called with `args`, into self.state.forward_result

Basically follows this format:

 out_target <- λ(Args1 Args2 Args3) ...

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
				(if (eql (cl-waffe2/vm.generic-tensor:tensor-attribute var) :chain)
				    ""
				    "<Input>"))
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

;; The Goal:
;; [Move(Save-For-Backward)]
;; [Move(Deleted)]
;; [Sin]
;; [Move(Save-For-Backward)]
;; [Move(Deleted)]
;; ...
;;

;; Time(s)   |   Instruction ( * - Beyonds the average execution time)
;;0.007294   | <WfInst[Compiled: MoveTensorNode(SAVE_FOR_BACKWARD)] : TID1082631 <= op(TID1082631(128 128) <Param>TID1082615(128 128))>
;;0.005084   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1082620 <= op(TID1082620(128 128) <Param>TID1082615(128 128))> <- Should be deleted!
;;0.289967*  | <WfInst[Compiled: SINNODE-LISPTENSOR]                : TID1082620 <= op(TID1082631(128 128) TID1082620(128 128))>
;;0.006844   | <WfInst[Compiled: MoveTensorNode(SAVE_FOR_BACKWARD)] : TID1082659 <= op(TID1082659(128 128) TID1082620(128 128))>
;;0.005356   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1082648 <= op(TID1082648(128 128) TID1082620(128 128))> <- Should be deleted!
;;0.262948*  | <WfInst[Compiled: SINNODE-LISPTENSOR]                : TID1082648 <= op(TID1082659(128 128) TID1082648(128 128))>
;;0.006389   | <WfInst[Compiled: MoveTensorNode(SAVE_FOR_BACKWARD)] : TID1082687 <= op(TID1082687(128 128) TID1082648(128 128))>
;;0.005264   | <WfInst[Compiled: MOVETENSORNODE-CPUTENSOR]          : TID1082676 <= op(TID1082676(128 128) TID1082648(128 128))> <- Should be deleted!
;;0.219833*  | <WfInst[Compiled: SINNODE-LISPTENSOR]                : TID1082676 <= op(TID1082687(128 128) TID1082676(128 128))>

;; In the iseq above, nodes that should be inplace is following:
;;  ...
;; TID1082620 <= op(TID1082620, TID1082612)
;;
;; ...
;;
;; The algorithm that detects such a node is simple:
;;
(defun apply-in-place-mutation! (iseq leaves)
  (declare (type list iseq leaves))

  (when (not *no-grad*)
    (return-from apply-in-place-mutation!))
  
  (let ((ref-table (make-hash-table)))

    ;; First, Register all tensors appeared in the computation node
    (mapc
     #'(lambda (variable)
	 ;; Tensors that can be destructed is:
	 ;; InputTensor (set=0)

	 (if (eql (tensor-attribute variable) :chain)
	     (setf (gethash (tensor-id variable) ref-table) 0)
	     (setf (gethash (tensor-id variable) ref-table) nil))) ;; attirubute=:input -> The tensor is parameter/input data, so never destruct it.
     leaves)

    ;; Nodes are the sequence of:
    ;; OUT_PLACE <- OP(ARG1, ARG2, ARG3, ...)
    ;; Usually, OUT_PLACE appears in the ARGn (e.g.: out <- op(x, out))

    ;; Here, we start reading the iseq from the bottom, and register all tensors that appeared at `ref-table`.
    ;; For the first appearance, the number in the ref-table should be 0, and also it is the equivalent to "JUST READING THE LAST USE OF TENSOR"
    ;; So the latest MoveTensor related to this, should be ignored.
    ;; If the count is set to >= 1 or nil, the MoveTensorNode is needed (also, we have to pay attention for movetensor-save-for-backward parameter)
    ;; Which indicatest the call of MoveTensorNode is intended, and shouldn't be ignored.

    (loop for instruction in iseq
	  for nth fixnum upfrom 0 do
	    (let* ((move-p (movetensor-p (wfop-node instruction)))
		   (next-node (nth (1+ nth) iseq))
		   (next-sv4bw-p (and next-node
				      (movetensor-p (wfop-node next-node))
				      (cl-waffe2/base-impl:mv-lazy-sv4bw (wfop-node next-node))))
		   (node   (wfop-node instruction)))

	      ;; ^ Reading in this way:
	      ;; | [MoveTensorNode] X   <- X, Z
	      ;; |    [SinNode]     OUT <- (OUT, X) ... Reading the value of SinNode, we decide whether delete above MoveTensorNode or not.
	      
	      ;; Instruction: OUT <- OP(out, arg1 arg2, ...)
	      ;; Counting up the number of reference counts, arg1, arg2 ... += 1
	      ;; In this case, we set `out` = 0, because the result overwrites the `out`

	      (when (and
		     move-p
		     ;; As far as I remember, this condition is intended that the copying for re-aranging memory-layout?
		     ;; But it could be possible to delete it, and delete more MoveTensorNode, especially for viewed tensor.
		     (apply #'cl-waffe2/vm.generic-tensor::order-reductable-p 0 (wfop-args instruction)) ;; <- I dunno if is it worth it? should be tested
		     (not (tensor-protect-me (car (wfop-args instruction))))
		     (not (cl-waffe2/base-impl:mv-lazy-sv4bw (wfop-node instruction))) ;; not save for backward
		     (not (movetensor-save-for-backward node)) ;; ... :force t
		     )
		;; Invoking this form == The MoveTensorNode can be deleted as long as ref-table condition is ok.
		;; If the instruction corresponds with MoveTensorNode or MoveScalarTensorNode
		;; Before counting up the reference, we judge whether MoveTensor is needed.

		(let (;;(place  (car (wfop-args instruction)))
		      (target (second (wfop-args instruction))))
		  ;; MoveTensorNode: out <- out, tensor_to_be_copied
		  ;; But with ignored:
		  ;; [Deleted] : out <- _, tensor_to_be_copied ;; <- _ is never allocated

		  ;; The condition to become [deleted] is:
		  ;; tensor_to_be_copied is the last reference in the computation node
		  (when (and
			 (numberp (gethash (tensor-id target) ref-table))
			 (or (= 0 (gethash (tensor-id target) ref-table))
			     (= 1 (gethash (tensor-id target) ref-table)))
			 (not next-sv4bw-p)
			 )
		    ;; Replace op with the lambda function just returning y

		    ;;(print "SET:")
		    ;;(print instruction)
		    (setf (wfop-op instruction) #'(lambda (x y)
						    (declare (ignore x))
						    y))
		    (setf (movetensor-ignore-me (wfop-node instruction)) t))))

	      ;; Counting up the ref-table
	      ;; OUT <- OP(arg1, arg2, arg3)

	      ;; OUT ... set ref-n=0 because the value is overwritten by OP
	      ;; arg1 arg2 arg3 ... set +=1 as long as registered in the table.

	      (mapc #'(lambda (arg)
			(when (numberp (gethash (tensor-id arg) ref-table))
			  (incf (gethash (tensor-id arg) ref-table)
				(if (movetensor-p (wfop-self instruction))
				    2
				    1))))
		    (wfop-args instruction))

	      (let ((out-id (tensor-id (wfop-self instruction))))
		(when (numberp (gethash out-id ref-table))
		  (setf (gethash out-id ref-table) 0))))))
  nil)



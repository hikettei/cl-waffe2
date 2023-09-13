
(in-package :cl-waffe2/vm)


;; [TODO] Add WfInstrucion to the docstring
;; WfInstruciton = cl-waffe2 IR

(defstruct (WfInstruction
	    (:conc-name wfop-)
	    (:constructor make-wfop (op self node args &key (sv4bw nil) (out-to nil) (fuse-prev nil) (block-iseq nil) (fused-body-cache nil) (call-with-view nil))))
  "
## [struct] WfInstruction

WfInstruction is IR for cl-waffe2 VM. Its form is represented by an extended Wengert list which is able to return multiple outputs. In this document, we call this *cl-waffe2 IR*, and compiled cl-waffe2 code, that is, the list of WfInstruction is called **InstructionSeq** or **iseq**. Unlike other frameworks, this IR is not only used to represent backpropagation but also forward propagation.


In cl-waffe2, WfInstruction is created by compiling AbstractNode, and its operation can be obtained by compiling lisp code or passing lambda functions.

A single WfInstruction represents:

```
out_to[0], out_to[1], ... <- λ(Args[0], Args[1], Args[2], ...)
 ^wfop-out-to                 ^wfop-op     ^wfop-args
```

where λ represents the operation. And, if any, `ArgsN` is wrapped with `SV4BW`.

```
out_to[0], out_to[1], ... <- λ(SV4BW(Args[0]), Args[1], Args[2], ...)
                                  ^ wfop-sv4bw
```

SV4BW (i.e: save-for-backward) is a temporary tensor to compute backwards and cl-waffe2 reads the `:save-for-backward` slots in the `define-impl` macro, and the corresponding tensors are copied.

### Slots

`wfop-op[function]` corresponds with compiled λ function.

`wfop-node[AbstractNode or string or function]` The node which generates λ function. For the most case, this slot is set to `AbstractNode`, but the node is something special, (e.g.: `CodeBlock`, `IfNode` etc...), set to `function`.

`wfop-out-to[list of AbstractTensor]` indicates list of tensors that results are to be stored.

`wfop-self[AbstractTensor]` corresponds with `out_target`, that is, the tensor to store the results

`wfop-args[list of AbstractTensor]` corresponds with `(tensor-variable wfop-self)`. tensors to be called with: `arg1 arg2 arg3...`.

`wfop-sv4bw[list of AbstractTensor]` indicates list of tensors storing save-for-backward tensors. if the corresponding position is `save-for-backward=nil`, the corresponding position also become nil.
"
  (op   op   :type function)  
  (node node :type (or function string null AbstractNode))
  (out-to    out-to :type list)
  (self self :type AbstractTensor)
  (block-iseq nil :type list)
  (args args :type list)
  (sv4bw sv4bw :type list)
  (error-check-p nil :type boolean) ;; Indicates the first shape-inspection has done?
  (bw-is-leaf-p nil :type boolean)
  (call-with-view call-with-view :type (or null cl-waffe2/vm.generic-tensor::Ranked-Loop))
  (fuse-prev fuse-prev :type (or null list))
  (fused-body-cache fused-body-cache :type (or null list)))

;; (defstruct (Composable-Operator <- separate call-with-view from body
;; (defun .cop (cop1 cop2) ...)

(defparameter *omit-args-n* 5)
(defparameter *opname-indent-to* 0 "Adds a space for this param times")
(defparameter *no-newline* nil)

(defmethod print-object ((inst WFInstruction) stream)
  (let ((ignored-p (and (movetensor-p (wfop-node inst))
			(movetensor-ignore-me (wfop-node inst)))))
    (format stream
	    "~a~a : ~a<= op(~a)>~a"
	    (instruction-opname inst)
	    (with-output-to-string (out)
	      (dotimes (i (- *opname-indent-to* (length (instruction-opname inst)))) (princ " " out)))
	    (with-output-to-string (out)
	      (dolist (o (wfop-out-to inst))
		(if (slot-value o 'cl-waffe2/vm.generic-tensor:requires-grad)
		    (princ "<Param>" out)
		    (when (not (eql (cl-waffe2/vm.generic-tensor:tensor-attribute o) :chain))
		      (princ "<Input>" out)))		      
		(princ (tensor-id o) out)
		(princ " " out)))
	    (if ignored-p
		(with-output-to-string (out)
		  (format out "~a{~(~a~), ~a}" (tensor-id (second (wfop-args inst))) (dtype (second (wfop-args inst))) (shape (second (wfop-args inst)))))
		(if (>= (length (wfop-args inst)) *omit-args-n*)
		    (format nil "..., x~a,..." (length (wfop-args inst)))
		    (with-output-to-string (out)
		      (dotimes (i (length (wfop-args inst)))
			(let ((var (nth i (wfop-args inst)))
			      (sv4 (and (not *no-grad*) (nth i (wfop-sv4bw inst)))))
			  (format out "~a~a~a{~(~a~), ~a}~a~a" ;;
				  (if (slot-value var 'cl-waffe2/vm.generic-tensor::requires-grad)
				      "<Param>"
				      (if (eql (cl-waffe2/vm.generic-tensor:tensor-attribute var) :chain)
					  ""
					  "<Input>"))
				  (if sv4
				      "SV4BW("
				      "")
				  (tensor-id var)
				  (dtype var)
				  (shape var)
				  (if sv4
				      ")"
				      "")
				  (if (nth (1+ i) (wfop-args inst))
				      " "
				      "")))))))
	    (if *no-newline*
		""
		(format nil "~%")))))

(defmethod instruction-opname ((inst WFInstruction))
  (format nil "<WfInst[op=~a]"
	  (if (functionp (wfop-node inst))
	      (funcall (wfop-node inst))
	      (if (movetensor-p (wfop-node inst))
		  (if (movetensor-ignore-me (wfop-node inst))
		      "Setq{Pruned}"
		      (if (cl-waffe2/base-impl:mv-lazy-sv4bw (wfop-node inst))
			  (if (scalar-p (wfop-self inst))
			      "MoveScalarNode(SAVE_FOR_BACKWARD)"
			      "MoveTensorNode(SAVE_FOR_BACKWARD)")
			  (class-name (class-of (wfop-node inst)))))
		  (class-name (class-of (wfop-node inst)))))))


(defmethod instruction-opname-table ((inst WFInstruction))
  (cl-ppcre:regex-replace-all
   "(\\n|\\s*$)"
   (format nil "~a"
	   (if (functionp (wfop-node inst))
	       (funcall (wfop-node inst))
	       (if (movetensor-p (wfop-node inst))
		   (if (movetensor-ignore-me (wfop-node inst))
		       "Setq{Pruned}"
		       (if (cl-waffe2/base-impl:mv-lazy-sv4bw (wfop-node inst))
			   (if (scalar-p (wfop-self inst))
			       "MoveScalarNode(SAVE_FOR_BACKWARD)"
			       "MoveTensorNode(SAVE_FOR_BACKWARD)")
			   (class-name (class-of (wfop-node inst)))))
		   (class-name (class-of (wfop-node inst))))))
   ""))

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

;; reverse-iseq .. Set T if the node is compiled as a toplevel of reverse mode
(defun apply-in-place-mutation! (iseq leaves &key (reverse-iseq nil))
  (declare (type list iseq leaves)
	   (type boolean reverse-iseq)
	   (optimize (speed 3)))

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
    ;; OUT_PLACE1 OUT_PLACE2 <- OP(ARG1, ARG2, ARG3, ...)
    ;; Usually, OUT_PLACE appears in the ARGn (e.g.: out <- op(x, out))

    ;; Here, we start reading the iseq from the bottom (as the same order of reverse mode), and register all tensors that appeared at `ref-table`.
    ;; For the first appearance, the number in the ref-table should be 0, and also it is the equivalent to "JUST READING THE LAST USE OF TENSOR"
    ;; So the latest MoveTensor related to this, should be ignored.
    ;; If the count is set to >= 1 or nil, the MoveTensorNode is needed (also, we have to pay attention for movetensor-save-for-backward parameter)
    ;; Which indicatest the call of MoveTensorNode is intended, and shouldn't be ignored.

    (loop for instruction in iseq
	  for nth fixnum upfrom 0 do
	    (let* ((move-p    (movetensor-p (wfop-node instruction)))
		   (node      (wfop-node instruction)))

	      ;; ^ Reading in this way:
	      ;; | [MoveTensorNode] X   <- X, Z
	      ;; |    [SinNode]     OUT <- (OUT, X) ... Reading the value of SinNode, we decide whether delete above MoveTensorNode or not.
	      
	      ;; Instruction: OUT <- OP(out, arg1 arg2, ...)
	      ;; Counting up the number of reference counts, arg1, arg2 ... += 1
	      ;; In this case, we set `out` = 0, because the result overwrites the `out`

	      ;; When found MoveTensorNode:
	      (when (and
		     move-p
		     ;; As far as I remember, this condition is intended that the copying for re-aranging memory-layout?
		     ;; But it could be possible to delete it, and delete more MoveTensorNode, especially for viewed tensor.
		     (not (tensor-protect-me (second (wfop-args instruction))))
		     (not (movetensor-save-for-backward node)) ;; when :force=t, never deleted.
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
		  (when (or reverse-iseq
			    (and
			     (numberp (gethash (tensor-id target) ref-table))
			     (= 0 (the fixnum (gethash (tensor-id target) ref-table)))))
		    ;; Replace op with the lambda function just returning y

		    ;; The target is allocated:
		    (if (and reverse-iseq			     
			     (cl-waffe2/vm.generic-tensor::vec (car (wfop-args instruction))))
			(setf (wfop-op instruction)
			      #'(lambda (x y)
				  (setf (tensor-vec x) (tensor-vec y))
				  y))
			(setf (wfop-op instruction) #'(lambda (x y)
							(declare (ignore x))
							y)))
		    (setf (movetensor-ignore-me (wfop-node instruction)) t))))

		;; Counting up the ref-table
		;; OUT <- OP(arg1, arg2, arg3)

	      ;; OUT ... set ref-n=0 because the value is overwritten by OP
	      ;; arg1 arg2 arg3 ... set +=1 as long as registered in the table.
	      
	      (mapc #'(lambda (arg)
			(when (and
			       move-p
			       (numberp (gethash (tensor-id arg) ref-table)))
			  (incf (the fixnum (gethash (tensor-id arg) ref-table)))))
		    (if (and reverse-iseq move-p (= (length (wfop-args instruction)) 3))
			(cdr (wfop-args instruction))
			(wfop-args instruction)))

	      (let ((out-to (wfop-out-to instruction)))
		(dolist (out out-to)
		  (when (numberp (gethash (tensor-id out) ref-table))
		    (setf (gethash (tensor-id out) ref-table) 0)))))))
  nil)

(defun %in-place-vm-ops! (iseq)
  ;; Make ViewTensorNode/ReshapeNode/PermuteNode In-place
  ;; Iseq ... Follows the execution order
  (declare (type list iseq))
    
  (loop for inst of-type WfInstruction in iseq do
    ;; ViewTensorNode: Result Base -> Result
    ;; Set Result.id <- Base.id
    (when (or (typep (wfop-node inst) 'cl-waffe2/base-impl::ViewTensorNode))
      (iseq-update-tensor-name! iseq
				(tensor-id (second (wfop-args inst)))
				(tensor-id (car    (wfop-args inst)))))

    ;; Reshape, Permute: Result Base -> Base
    ;; Base.id <- Result.id
    (when (or (typep (wfop-node inst) 'cl-waffe2/base-impl::ReshapeTensorNode)
	      (typep (wfop-node inst) 'cl-waffe2/base-impl::Permute-Node))
      (iseq-update-tensor-name! iseq
				(tensor-id (car    (wfop-args inst)))
				(tensor-id (second (wfop-args inst)))))))





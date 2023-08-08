
(in-package :cl-waffe2/vm)


;;(defstruct InstructionSeq)

;; build関数 ... 計算のーどを辿って 一次元であるInstructionSeqを生成する
;; forward/backward用にそれぞれ作る
;; forward/backward関数 ... InstructionSeqを辿って色々する

;; funcall多用する・・・

(defstruct (WFInstruction
	    (:conc-name wfop-)
	    (:constructor make-wfop (op self node args)))
  "
## [struct] WFInstruction
"
  (op   op   :type function)
  (node node :type (or null AbstractNode))
  (self self :type AbstractTensor)
  (args args :type list))

;; (defstruct (Composable-Operator <- separate call-with-view from body
;; (defun .cop (cop1 cop2) ...)

(defmethod print-object ((inst WFInstruction) stream)
  (format stream
	  "<WfInst[Compiled: ~a] : ~a => ~a>~%"
	  (class-name (class-of (wfop-node inst)))
	  (tensor-id (wfop-self inst))
	  (with-output-to-string (out)
	    (dolist (var (wfop-args inst))
	      (format out "~a~a " (tensor-id var) (shape var))))))

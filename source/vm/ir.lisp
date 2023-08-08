
(in-package :cl-waffe2/vm)


;;(defstruct InstructionSeq)

;; build関数 ... 計算のーどを辿って 一次元であるInstructionSeqを生成する
;; forward/backward用にそれぞれ作る
;; forward/backward関数 ... InstructionSeqを辿って色々する

;; funcall多用する・・・

(defstruct (WFInstruction
	    (:conc-name wfop-)
	    (:constructor make-wfop (op node args)))
  "
## [struct] WFInstruction
"
  (op   op   :type function)
  (node node :type AbstractNode)
  (args args :type list))

(defmethod print-object ((inst WFInstruction) stream)
  (format stream
	  "<<WFInstruction
    Op:~a
    Node:~a
    Args: ~a
>>"
	  (wfop-op inst) 
	  (wfop-node inst)
	  (wfop-args inst)))




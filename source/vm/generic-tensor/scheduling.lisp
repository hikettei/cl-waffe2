
(in-package :cl-waffe2/vm.generic-tensor)

;; NFA <-> DFA
;; DFA --> NFA

;; out-tensorからTraceして、ノードに含まれる ignore-me optionをTにするか、
;; 遷移先(variables)を適度置き換える

;; deterministic-p
;; nondeterministic-p

;; Speed-Majorな最適化と (First)
;; Memory-Majorな最適化のアルゴリズムがある (Second)

;; Speed-Major
;; Tensorで分岐しているノードをCopyして依存関係をなくす
;; 非決定的な計算ノードに直してlparallelで並列化する


;; 計算木の各部分は、一つのTensor-Inputに依存する
;; この依存を解決するには
;; NFA -> DFAに変換 (Memory-Major)
;; TensorをCopy (Speed-Major)

(defun deterministic-p (tensor)
  "Returns t if tensor's node is deterministic
[Any-Previous-Node]
    |
[AnyNode] <- The Given Tensor
    |
"
  (declare (type AbstractTensor tensor))
  (= (length (tensor-variables tensor)) 1))

(defun non-deterministic-p (tensor)
  "Returns t if tensor's node is non-deterministic
[Node1] [Node2] ...
    |------|
[AnyNode] <- The Given Tensor
    |
"
  (declare (type AbstractTensor tensor))
  (> (length (tensor-variables tensor)) 1))

(deftype node-state-t ()
  `(member :deterministic :non-deterministic))

(declaim (ftype (function (AbstractTensor) node-state-t) node-state))
(defun node-state (tensor)
  (if (deterministic-p tensor)
      :deterministic
      :non-deterministic))

(defun movetensor-p (tensor)
  (typep tensor 'MoveTensorNode))

(defun optimize-computation-node! (out-tensor major n-cores)
  "The function optimize-computation-node! do these works:

1. Optimize MoveTensorNode
2. Optimize the connection of ChainTMP
3. Scheduling the lparallel depending on their nodes and threads."
  (declare (type AbstractTensor out-tensor)
	   (type fixnum n-cores)
	   (type (and keyword (member :speed :memory)) major))
  ;; <<TODO: Assert -> out-tensor is NOT MoveTensorNode>>


  ;;    [MoveTensor]
  ;;         |
  ;;    [AddTensor]
  ;;         |
  ;;    [MoveTensor] <- our work is to distinguish what MoveTensor is needed and ignore them.
  ;;         |
  ;;    [AddTensor]
  ;;

  (let* ((current-node   (tensor-backward out-tensor))
 	 (past-variables (tensor-variables out-tensor))
	 (state (node-state out-tensor)) ;; deterministic or non-deterministic
	 )
    
    ;; e.g.:
    ;;    Variables       | current-node
    ;; [MoveTensor], [AddNode] -> [CurrentNode]
    ;;  PastVar1      PastVar2      out-tensor (Corresponding tensors)

    ;; If the tensor is deterministic and the higher node is MoveTensor
    ;; Here's no dependencies between CurrentNode and N Higher Node.
    ;; Therefore no need to make a copy, ignore it.
    ;; Note that The case when the higher node's tensor is a Input.
    ;; In that case, No Side-Effects are ALLOWED.

    ;; e.g.: [InputTensor] -> [MoveTensor] -> [1DFunction]
    ;; ↑ Copy it

    (when (and (eql state :deterministic))

      )
    
   
    

    ))


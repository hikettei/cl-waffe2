
(in-package :cl-waffe2/vm.iterator)

(defstruct Reference
  "A[stride * idx + offset] where idx = range[0], range[1], ..."
  (offset)
  (size)
  (stride))

(defstruct Action
  "Action = target_tid[range(from, to, by)]"
  (tid  nil :type symbol)
  (ref  nil :type Reference))

(defstruct Invocation
  "sources op targets"
  (op nil :type symbol)
  (sources nil :type list)
  (targets nil :type list))

(defun trace-invocation (op source-tensors target-tensors &key (kernel-rank 1) (collapse t))
  "call-with-view -> invocation
Assertion: Shapes are already determined."
  (assert (= kernel-rank 1) () "kernel-rank must be 1")
  (let ((result))
    (wf/t::do-compiled-loop*
	(solve-loop-order `(,@source-tensors ,@target-tensors) kernel-rank (not collapse) :mode :runtime)
      #'(lambda (&rest views)
	  (push
	   (make-invocation
	    :op op
	    :sources
	    (loop for n upfrom 0 below (length source-tensors)
		  collect
		  (make-action
		   :tid (tensor-id (nth n source-tensors))
		   :ref (make-reference
			 :offset
			 (offset-of (nth n views) 0)
			 :size
			 (size-of   (nth n views) 0)
			 :stride
			 (stride-of (nth n views) 0))))
	    :targets
	    (loop for n upfrom 0 below (length target-tensors)
		  collect
		  (let ((k (+ n (length source-tensors))))
		    (make-action
		     :tid (tensor-id (nth n target-tensors))
		     :ref (make-reference
			   :offset
			   (offset-of (nth k views) 0)
			   :size
			   (size-of   (nth k views) 0)
			   :stride
			   (stride-of (nth k views) 0))))))
	   result))
      `(,@source-tensors ,@target-tensors))
    (reverse result)))

(defun read-reference (reference)
  "-> (values iter-size reduce-p)"
  (declare (type reference reference))
  (values
   (reference-size reference)
   (= 0 (reference-stride reference))))

(defun read-action (action)
  (declare (type action action))
  (multiple-value-bind (sizes reduce-p) (read-reference (action-ref action))
    (values sizes reduce-p (action-tid action))))

(defun read-invocation (invocation)
  (declare (type invocation invocation))
  (flet ((reading (x)
	   (multiple-value-list (read-action x))))
    (values
     (invocation-op invocation)
     (map 'list #'reading (invocation-sources invocation))
     (map 'list #'reading (invocation-targets invocation)))))	    

;; !! topi的な命令と並行してJITをする。 (Dynamic Shape)
;; (Forward ...)ってやると，Dynamic Shapeが全部決定してるから，Sessionのリスト作ってJITしてcacheができる

;; merge range [0 3] [3 6] -> [0 6]
;; 1. cl-waffe2IRにコンパイルする
;; 2. (cl-waffe2/vm) Subscript DSLから，ElementWiseと load-pointer関連だけ引っ張ってくる
;; 3. Dynamic-Shapeの状態ごとにコンパイルをCacheするJITを並行で実装 (JITCPUTensor, MetalTensor) } <- ここの管理状態を追加
;; define-impl-jitを追加，invoke-sessionをする (これでJIT可能を宣言 + 自動化)

(defstruct Session
  (invocation nil :type invocation)
  (ops nil :type list))

(defun ensure-invocations (inv1 inv2)
  (declare (type Invocation inv1 inv2))
  ;; sessions that can be fused:
  ;; ops does not matter, but the size of iters
  (multiple-value-bind (op1 s1 t1) (read-invocation inv1)
    (declare (ignore op1))
    (multiple-value-bind (op2 s2 t2) (read-invocation inv2)
      (declare (ignore op2))
      (flet ((satisfier (act1 act2)
	       (and
		(equal (car act1) (car act2)))))
	(and
	 (every #'satisfier s1 s2)
	 (every #'satisfier t1 t2))))))

(defun merge-session! (session1 invocation)
  "Attemptes to Session1 <- Session2"
  (declare (type Session session1)
	   (type Invocation invocation))
  (if (ensure-invocations (session-invocation session1) invocation)
      (progn
	(setf (session-ops session1) `(,@(session-ops session1) ,invocation))
	t)
      nil))

(defun initialize-session (invocation)
  (make-session
   :invocation invocation
   :ops nil))

(defun equivalent-invocations-p (inv1 inv2)
  "Sizes does not matter; Returns T if Invocation is produced from the equivalent operations."
  (multiple-value-bind (op1 s1 t1) (read-invocation inv1)
    (multiple-value-bind (op2 s2 t2) (read-invocation inv2)
      (flet ((satisfier (act1 act2)
	       (and
		(equal (second act1) (second act2))
		(equal (third  act1) (third act2)))))
	(and
	 (eql op1 op2)
	 (every #'satisfier s1 s2)
	 (every #'satisfier t1 t2))))))
    
(defun fuse-equivalents (pc invocations)
  "Return (list same-invocations new-pc)"
  (let ((new-pc pc)
	(latest (nth pc invocations))
	(result))
    (loop named collector
	  for p upfrom pc below (length invocations) do
	    (if (equivalent-invocations-p latest (nth p invocations))
		(progn
		  (incf new-pc)
		  (push (nth p invocations) result))
		(return-from collector))) 
    (values
     (reverse result)
     new-pc)))

(defun make-table-by-size (invocations)
  (let ((table (make-hash-table :test #'equal)))
    (dolist (invc invocations)
      (multiple-value-bind (op s1 t1) (read-invocation invc)
	(declare (ignore op))
	(let ((key `(,@(map 'list #'car s1) ,@(map 'list #'car t1))))
	  (if (gethash key table)
	      (push invc (gethash key table))
	      (setf (gethash key table) (list invc))))))
    (maphash
     #'(lambda (k v)
	 (setf (gethash k table) (reverse v)))
     table)
    table))

(defun select-keys (latest-key table)
  (let ((keys (hash-table-keys table)))
    (butnil
     (delete-duplicates
      `(,(find latest-key keys :test #'equal)
	,@keys)
      :test #'equal))))

(defun solve-invocations (invocations &aux (pc 0))
  "Fuses several invocations into a single kernel.
Sizeが同じIterationは一回のSessionにまとめる。
時系列の依存関係を維持する
Reduction ( stride=0 に足し合わせる ) は一時領域が必要 (since shared)

同じ命令(same op, same tensor ids)は順序を入れ替えてもOK

Input:  (list invocation)
Output: (list Session)
"
  (let ((sessions)
	(latest-key))
    (loop  named session-fuse while t do
      (when (null (nth pc invocations)) (return-from session-fuse))
      (multiple-value-bind (targets new-pc) (fuse-equivalents pc invocations)
	(setf pc new-pc)
	(let* ((target-by-size (make-table-by-size targets))
	       (key-order      (select-keys latest-key target-by-size)))
	  ;; Filter by iter sizes (and they can be fused into a single kernel.)
	  (dolist (key key-order)
	    (let ((invcs (gethash key target-by-size)))
	      (dolist (invc invcs)
		(when (null sessions)
		  (push (initialize-session invc) sessions))
		(if (merge-session! (car (last sessions)) invc)
		    nil
		    (progn
		      (push (initialize-session invc) sessions))))))
	  (setf latest-key (car (last key-order))))))
    (print (length sessions))
    (print sessions)
    (reverse sessions)))


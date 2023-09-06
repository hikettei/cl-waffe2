
(in-package :cl-waffe2/vm)

;; [TODO] memory-pool.lispを削除
;; [TODO] Dynamically-Shapingを安定化する
;; [TODO] Stride ... In-placeの保証のもとに数値を優先

;; [TODO] モデルをコンパイルするとToplevelではallocation-state構造が帰ってくる
;; [TODO] 上をグローバル変数にセットしてその下で計算をしないといけない

;; [TODO] REPL上ではどうする？今までののこす？
;; [TODO] 局所性の最適化 -> 後で
;; [TODO] DtypeとかDeviceが違うTensorは同じにしたらダメ


;; [TODO] 抜ける時にGlobalのMemory-Poolに移動しないと
;; with-static-allocationの外でProceedで繋げることができない。
;; defparameterでglobalなallocationを宣言しておく？
;; vecに格納しておく
;; (grad tensor) <- 読み込める？

;; [TODO] AdjustableSymbolの管理 ... VMAllocationに任せる？
;; Adjustable-Shape VMAllocationがないとできない <-
;; Locality 
;; Eliminate-Undetermined-XXX <- osoi

;; コンパイル時と実行時のTensorIDは一致していないかも・・・

(defun tensor-tmp-p (tensor)
  "Returns T if the given tensor is subject to be optimized locality"
  (declare (type AbstractTensor tensor))
  (and (eql     (tensor-facet tensor) :input)
       (not     (scalar-p tensor))
       (stringp (tensor-name tensor))))

(defstruct (VMAllocation
	    (:conc-name vmalloc-))
  "
## [struct] VMAllocation

Records the statue and result of localized allocation state by cl-waffe2 VM.
"
  (allocated-p nil :type boolean)
  (id-routes  (make-hash-table) :type hash-table)
  (id->tensor (make-hash-table) :type hash-table)
  (reduce-rate 0.0 :type single-float))

(defun read-from-table (id allocation)
  (declare (type VMAllocation allocation))
  (gethash
   (gethash id (vmalloc-id-routes allocation))
   (vmalloc-id->tensor allocation)))

(defmethod print-object ((model VMAllocation) stream)
  (format stream "{VMAllocation:
    memory-pool=~a,
    reduce-rate=~a,
    routes:
~a}"
	  (vmalloc-id->tensor model)
	  (vmalloc-reduce-rate model)
	  (with-output-to-string (out)
	    (maphash #'(lambda (k v)
			 (format out "        ~a -> ~a~%" k v))
		     (vmalloc-id-routes model)))))

;; [TODO] Memory-Poolのごにゃごにゃと合わせて
;; [TODO] Toplevelの引数になるTensorはset-inputしないといけない e.g.: (make-input ... :X)
(defun adjust-allocation! (allocation shape-table)
  "
## [function] adjust-allocation!

Dynamically Shaped Tensors registered in the `allocation`, is able to form their own shapes by calling this function.
Reading the value of shape-table: (e.g.: A->1, B->2 ...), adjusts the size of storage vector allocated in the allocation.
If the re-allocation is performed, frees the old one.
"
  (declare (type VMAllocation allocation)
	   (type hash-table shape-table)
	   (optimize (speed 3)))

  (flet ((->num (val) (the number (if (numberp val) val (gethash val shape-table)))))
    (loop for tensor being the hash-values in (vmalloc-id->tensor allocation)
	  ;; If the tensor is DYNAMYCALLY SHAPED:
	  if (some #'symbolp (cl-waffe2/vm.generic-tensor::tensor-input-shape tensor))
	    do (if (>= (the fixnum (apply #'* (map 'list #'->num (cl-waffe2/vm.generic-tensor::original-shape tensor))))
		       (apply #'* (map 'list #'->num (cl-waffe2/vm.generic-tensor::tensor-input-shape tensor))))
		   ;; The storage size is enough, Keep Using a old one:
		   (setf (slot-value tensor 'cl-waffe2/vm.generic-tensor::orig-shape)
			 (map 'list #'->num (cl-waffe2/vm.generic-tensor::tensor-input-shape tensor)))

		   ;; Update the allocation:
		   (when (not (scalar-p tensor))
		     ;; Update
		     (funcall (the function (tensor-finalizer tensor)))
		     (setf (tensor-vec tensor) nil)
		     (setf (slot-value tensor 'cl-waffe2/vm.generic-tensor::orig-shape)
			   (map 'list #'->num (cl-waffe2/vm.generic-tensor::tensor-input-shape tensor)))
		     (setf (tensor-vec tensor) (make-tensor
						(cl-waffe2/vm.generic-tensor::original-shape tensor)
						:dtype (dtype tensor)
						:order (order tensor)
						:device (class-of tensor))))))))

(defun maybe-allocate! (allocation)
  (declare (type VMAllocation allocation))
  (when (not (vmalloc-allocated-p allocation))
    (loop for tensor being the hash-values in (vmalloc-id->tensor allocation) do
      ;; Reading the orig-shape
      (setf (slot-value tensor 'cl-waffe2/vm.generic-tensor::orig-shape)
	    (map 'list #'cl-waffe2/vm.generic-tensor::read-symbol
		 (cl-waffe2/vm.generic-tensor::original-shape tensor)))
      (when (null (cl-waffe2/vm.generic-tensor::vec tensor))
	(let ((use
		(if (scalar-p tensor)
		    (make-tensor 0 :dtype (dtype tensor) :device (class-of tensor) :order (order tensor))
		    (make-tensor
		     (cl-waffe2/vm.generic-tensor::original-shape tensor)		       
		     :dtype (dtype tensor)
		     :order (order tensor)
		     :device (class-of tensor)))))
	  (setf (tensor-vec tensor) (cl-waffe2/vm.generic-tensor::vec use)))))
    (setf (vmalloc-allocated-p allocation) T)))
	     
(defun storage-vec-from-memory-pool (allocation tensor)
  "Reading the allocated state, `allocation`, the function returns a storage vec of tensor."
  (declare (type VMAllocation allocation)
	   (type AbstractTensor tensor))

  (when (null (gethash (tensor-id tensor) (vmalloc-id-routes allocation)))
    (if (cl-waffe2/vm.generic-tensor::vec tensor)
	(return-from storage-vec-from-memory-pool (cl-waffe2/vm.generic-tensor::vec tensor))
	(error "tensor-vec: Attempted to read the storage vec of [~a, ~a, ~a] but failed because the tensor wasn't tracked when compiling.
Allocation State:
~a" (tensor-id tensor) (class-of tensor) (shape tensor) allocation)))
  
  (let ((result (read-from-table (tensor-id tensor) allocation)))
    (when (null result) ;; <- Originated from compiler.
      (error "tensor-vec: The tensor ~a~a~a isn't registered to the memory-pool.
To use InputTensor as a cache, envolve the tensor into your node by any means, as an argument."
	     (tensor-id tensor) (class-of tensor) (shape tensor)))

    (setf (tensor-vec tensor) (cl-waffe2/vm.generic-tensor::vec result))
    (cl-waffe2/vm.generic-tensor::vec result)))

(defmacro with-static-allocation ((allocation) &body body)
  "
## [macro] with-static-allocation

Declares the static allocation state to use.
"
  `(let ((*static-alloc-state* ,allocation))
     (maybe-allocate! ,allocation)
     ,@body))


;; ~~ [Implementation] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; 1. Combining   AbstractNode, create computation nodes with dispatching multiple devices.
;; 2. O(|V|+|E|)  Topological Sorting the node for forward propagation
;; 3. O(n)        Optimize the node by deleting MoveTensorNode which is unnecessary.
;; 4. O(n)        Generating the InstructionSeq for the backward propagation.
;; 5. O(?)        Merging fw/bw iseq and optimize the locality of the memory
;; 6. With call-with-view, Inlining/Collapsing the computation of view and loop
;; 7. O(N * path) With defpath macro, users can apply FusionOps
;; 8. Tada~ Completed!
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; [TODO] Optimize Backward...
;; tensor-protect-dout <- これ最後の参照のdoutは破壊してもOK
;; InstructionSeqを一直線に並べてからApply-In-Place-Mutation!したい

(defun optimize-memory-locality! (iseq-fw iseq-bw)
  (declare (type list iseq-fw)
	   (type (or null list) iseq-bw))

  ;; iseq-fw ... NIL is NG
  ;; iseq-bw ... NIL is ok

  (let* ((cache-tensor-table (make-hash-table))
	 (alloc-route-table  (make-hash-table))
	 (id->tensor-table   (make-hash-table))

	 (sv4bw-tensor-table (make-hash-table))
	 
	 (iseq `(,@(loop for inst in (reverse iseq-fw)
			 if (null (wfop-block-iseq inst))
			   append (list inst)
			 else
			   append (wfop-block-iseq inst))
		 ,@(loop for inst in iseq-bw
			 if (null (wfop-block-iseq inst))
			   append (list inst)
			 else
			   append (wfop-block-iseq inst)))))

    ;; iseq ... flattened list of iseq
    ;; VM executes them in the order of iseq[0] iseq[1] ... iseq[n] where n = program_counter

    ;; Counting up all tensors (TMP Tensor) used in the node.
    ;; All we need is the first appearance in the node.
    (loop for inst in iseq do 
      (when (tensor-tmp-p (wfop-self inst))
	(setf (gethash (tensor-id (wfop-self inst)) cache-tensor-table) (wfop-self inst)))
      (mapc
       #'(lambda (arg)
	   (when (tensor-tmp-p arg)
	     (setf (gethash (tensor-id arg) cache-tensor-table) arg)))
       (wfop-args inst))
      
      (mapc
       #'(lambda (tensor)
	   (when tensor
	     (setf (gethash (tensor-id tensor) sv4bw-tensor-table) tensor)))
       (wfop-sv4bw inst)))

    ;; Iseqを正しい順番にSort
    ;; :FREEと:USINGをリアルタイムで管理
    ;; BACKWARD時はgngn
    ;; 検索はapply '#'* original-shapeベースで
    ;; SV4BWは逆伝播で一回用いられたらその場で破棄する
    ;; define-opのsave-for-backwardの扱い？
    ;; defmodel-asでwith-static-allocationがネストしたときの扱い・・・
    ;; 最初のallocateはrouteから参照しないと・・・MoveでPruneされた後のTensorもallocしちゃう
    ;;(simulate-memory-pool! iseq cache-tensor-table)

    
    ;; [TODO] 前後でメモリ使用量計算して性能を評価する
    ;; [TODO] mempool-idxを入れ替えて最適化する
    
    (loop for tensor being the hash-values in cache-tensor-table do
      (setf (gethash (tensor-id tensor) alloc-route-table) (tensor-id tensor))
      (setf (gethash (tensor-id tensor) id->tensor-table) tensor))

    (loop for tensor being the hash-values in sv4bw-tensor-table do
      (setf (gethash (tensor-id tensor) alloc-route-table) (tensor-id tensor))
      (setf (gethash (tensor-id tensor) id->tensor-table) tensor))

    (make-vmallocation
     :id->tensor id->tensor-table
     :id-routes  alloc-route-table)))

(defun inst-set-p (inst) (and (movetensor-p (wfop-node inst)) (movetensor-ignore-me (wfop-node inst))))

;; Blockの形は維持するけど、Node側でIn-place-mutation!をするんじゃなくて
;; Flattenにして破壊的にin-place-mutation!してBlockの形で返す

;; RNN ... define-impl-opで埋め込む？
;; SV4BWでAllocしたのは局所性云々とはどうでもいい
;; Backward ... In-place-mutation!しない
;; InlineしてIseqにしてからIn-place-mutation!する(doutを繋げる)
;; O(N^2)
;; Enhancement: IDの番地を人間が読みやすくする

(defun simulate-memory-pool! (iseq memory-pool)
  (declare (optimize (speed 3))
	   (type list iseq)
	   (type hash-table memory-pool))
  
  (let ((adjustable-table  (make-hash-table :test #'equal))
	(fixed-shape-table (make-hash-table))
	
	(mempool-using-tensors nil)
	(pools nil)
	(pools-adj nil))
    (declare (type list pools pools-adj mempool-using-tensors))
    
    ;; 1. Sort Tensors by their storage size
    (maphash
     #'(lambda (key value)
	 (declare (ignore key))
	 (if (some #'symbolp (shape value))
	     (setf (gethash (shape value) adjustable-table) value)
	     (setf (gethash (apply #'* (shape value)) fixed-shape-table) value)))
     memory-pool)

    
    (labels ((args-last-ref-p (target-tensor pc)
	       ;; By the next time `target-tensor` is used as a `out-to`
	       ;; If the target-tensor isn't appeared in the wfop-args or reaches the end, return T
	       ;; Otherwise -> NIL
	       ;; (cdr (nthcdr ... <- do not include the current position
	       (loop for inst in (cdr (nthcdr pc iseq)) do
		 (if (find (the symbol (tensor-id target-tensor)) (wfop-args inst) :key #'tensor-id :test #'eql)
		     (return-from args-last-ref-p nil)
		     (if (find (the symbol (tensor-id target-tensor)) (wfop-out-to inst) :key #'tensor-id :test #'eql)
			 (return-from args-last-ref-p t)
			 nil))) ;; Keep exploring
	       ;; Reached the last -> T
	       t)
	     (find-from-pool (tensor)	     
	       (if (tensor-tmp-p tensor)
		   (if (some #'symbolp (shape tensor))
		       (find tensor pools-adj :test #'equal :key #'shape)
		       (find (the fixnum (apply #'* (shape tensor)))
			     pools
			     :test #'<=
			     :key #'(lambda (x) (apply #'* (shape x)))))))
	     (read-from-pool (tensor)
	       (when (find (the symbol (tensor-id tensor)) mempool-using-tensors)
		 (return-from read-from-pool tensor))
	       
	       (if (tensor-tmp-p tensor)
		   (let ((out (find-from-pool tensor)))
		     ;; Delete from pool
		     (when (null out)
		       (return-from read-from-pool tensor))
		     (push (tensor-id out) mempool-using-tensors)
		     (if (some #'symbolp (shape tensor))
			 (setq pools-adj (remove (the list (shape out)) pools-adj :test #'equal :key #'shape :count 1))
			 (setq pools
			       (remove (apply #'* (shape out))
				       pools
				       :test #'<=
				       :key #'(lambda (x) (apply #'* (shape x)))
				       :count 1))))
		   tensor))
	     (set-as-free (tensor)
	       (when (and (tensor-tmp-p tensor)
			  ;; The tensor is from memory-pool?
			  (find (the symbol (tensor-id tensor)) mempool-using-tensors))
		 (setq mempool-using-tensors (delete (tensor-id tensor) mempool-using-tensors :test #'eql))
		 (print mempool-using-tensors)
		 (if (some #'symbolp (shape tensor))
		     (push tensor pools-adj)
		     (push tensor pools)))))
      
      (loop for inst of-type WfInstruction in iseq for pc fixnum upfrom 0 do
	(let* ((args (if (inst-set-p inst)
			 (cdr (wfop-args inst))
			 (wfop-args inst)))
	       (args-last-p (map 'list #'(lambda (x) (args-last-ref-p x pc)) args))
	       (out-to      (wfop-out-to inst)))

	  ;; Allocation is required when:
	  ;;   The tensor is used as a out parameter

	  ;; Allocation is freed when:
	  ;;   args-last-p=T (Moved to the memory-pool)
	  
	  ;; WfInstruction: out-to[0], ... <- f(args[0], args[1], ...)


	  (mapc #'(lambda (tensor state)
		    (if state
			(set-as-free tensor)))
		args args-last-p)
	  
	  ;; ここの優先度？
	  ;; この最適化は合成しても一回呼び出しと等価？
	  (dolist (arg args)
	    (let ((result (read-from-pool arg)))
	      (setf (tensor-id arg) (tensor-id result))))
	  
	  (dolist (out out-to)
	    (let ((result (read-from-pool out)))	     
	      (setf (tensor-id out) (tensor-id result))
	      ))
	  


	  ))


      )))

;;(tensor-mempool-idx tensor) を更新する・・・

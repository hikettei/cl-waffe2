
(in-package :cl-waffe2/vm)


;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; [Paradigm of memory-pool with dynamically shaped tensors]
;; In cl-waffe2, memory-pool isn't global but local because everytime we call build, it also localized the
;; memory usage.

;;
;; - [build: toplevel] ----------
;; | X -> (A, B), Y -> (A, B)   |  <- Pool to use is declared by (with-static-allocation
;; | Pool = TMP1, TMP2          |     Symbols to use is declared by with-dynamically-shape-scope
;; ------------------------------
;;    |
;;    | Creating a new scope
;;    -------|- [build: %vm-move] ----------
;;           | A -> (R1, R2) B -> (R1, R2) | <- Similary, the new scope is created 
;;           | Pool = NIL                  |    The superior tensor's storage vec is filled with something
;;           ------------------------------|    So they never use %vm-move scope memory-pool
;;    |
;;    |
;;  (...) Keep Computing Somethings
;;

;; By copying all tensors in (Pool = ...), you can reuse the compiled iseq with different threads.
;; ^ defmodel-as :asif :node is implemented by it.
;; AbstractNode: f(lambda_fw, lambda_bw, tensors) -> g(tensors) where g is a thread-safe compiled program.
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; [TODO] defmodel-as ...  無駄なインライン化をして実装方式を増やさない
;;  -> Compiled-Composite自体をCacheする

;; [FIX] gradient-adderが増えてる
;; copy/delete-alloc テスト

;; [TODO] 上の表を実装してThread-safeにする
;; [TODO] %vm-moveをthread-safeに動かす defmodel-as ... thread-safe optionを追加
;; [TODO] gradient-adder naosu

;; [TODO] !softmax exp duplicated 削除
;; Global Memory-Poolの改善
;; define-op の SaveForBackwardの扱いは？テスト増やすべき

;; RNN ... define-impl-opで埋め込む？
;; SV4BWでAllocしたのは局所性云々とはどうでもいい
;; O(N^2)
;; Enhancement: IDの番地を人間が読みやすくする

;; define-opのsave-for-backwardの扱い？
;; defmodel-asでwith-static-allocationがネストしたときの扱い・・・
;; 最初のallocateはrouteから参照しないと・・・MoveでPruneされた後のTensorもallocしちゃう

;; memory-poolのmemory-poolが欲しい〜

;; Forward 動く？ Backwardが動かん〜
;; (!mul a b) AがInputTensorだとMoveTensorNodeを一つ減らせる

;; TODO Nested with-static-allocation

;; [TODO] DtypeのCastを計算ノードにする CastTensorNode-XXX
;; [TODO] ^ forward :aroundでデータ型の不一致を検知したら、outに従って自動でCasting

;; [TODO] memory-pool.lispを削除 (NO)
;; [TODO] Dynamically-Shapingを安定化する
;; [TODO] Stride ... In-placeの保証のもとに数値を優先

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
;; [BugFix] (!relu (!softmax a)))) backward
;; コンパイル時と実行時のTensorIDは一致していないかも・・・
;; allocationだけ新しくすればToplevelでAbstractNodeのコンパイルできる
;; -> (defun copy-allocation, delete-allocation

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
  (id2pool     (make-hash-table) :type hash-table)
  (reduce-rate 0.0 :type single-float))

(defmethod print-object ((model VMAllocation) stream)
  (format stream "{VMAllocation:
    id2pool=~a,
    reduce-rate=~a,
    allocated-p=~a
}"
	  (vmalloc-id2pool model)
	  (vmalloc-reduce-rate model)
	  (vmalloc-allocated-p model)))

;; Arguments (e.g.: created by (make-input `(A B) :X)) needs to be set-input
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

  (flet ((->num (val)
	   (the number
		(if (numberp val)
		    val
		    (or (gethash val shape-table)
			(error "adjust-allocation!: The symbol ~a is unknown. Choose from: ~a"
			       val
			       (alexandria:hash-table-keys shape-table)))))))
    (loop for tensor being the hash-values in (vmalloc-id2pool allocation)
	  ;; If the tensor is DYNAMYCALLY SHAPED:
	  if (some #'symbolp (cl-waffe2/vm.generic-tensor::tensor-input-shape tensor))
	    do (if (>= (the fixnum (apply #'* (map 'list #'->num (cl-waffe2/vm.generic-tensor::original-shape tensor))))
		       (apply #'*  (map 'list #'->num (cl-waffe2/vm.generic-tensor::tensor-input-shape tensor))))
		   ;; The storage size is enough, Keep Using a old one:
		   (setf (slot-value tensor 'cl-waffe2/vm.generic-tensor::orig-shape)
			 (map 'list #'->num (cl-waffe2/vm.generic-tensor::tensor-input-shape tensor)))

		   ;; Update the allocation:
		   (when (not (scalar-p tensor))
		     ;; Update
		     (funcall (the function (tensor-finalizer tensor)))
		     (setf (tensor-vec tensor) nil)
		     ;; Prev-allocation -> After-allocation
		     (setf (slot-value tensor 'cl-waffe2/vm.generic-tensor::orig-shape)
			   (map 'list #'->num (cl-waffe2/vm.generic-tensor::tensor-input-shape tensor)))
		     (setf (tensor-vec tensor)
			   (cl-waffe2/vm.generic-tensor::vec
			    (make-tensor
			     (cl-waffe2/vm.generic-tensor::original-shape tensor)
			     :dtype (dtype tensor)
			     :order (order tensor)
			     :device (class-of tensor)))))))))

(defun maybe-allocate! (allocation)
  (declare (type VMAllocation allocation))
  (when (not (vmalloc-allocated-p allocation))
    (loop for tensor being the hash-values in (vmalloc-id2pool allocation) do
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

(defun copy-allocate (allocation)
  "Makes a copy of given allocation and its storage vec is also copied so no thread-conflicts would happen."
  (declare (type VMAllocation allocation))

  (let ((allocation (copy-vmallocation allocation)))
    (setf (vmalloc-allocated-p allocation) NIL)
    (loop for key being the hash-keys      in (vmalloc-id2pool allocation)
	  for tensor being the hash-values in (vmalloc-id2pool allocation) do
	    (let ((use (if (scalar-p tensor)
			   (if (scalar-p tensor)
			       (make-tensor 0 :dtype (dtype tensor) :device (class-of tensor) :order (order tensor))
			       (make-clone tensor (tensor-name tensor))))))
	      (setf (slot-value use 'cl-waffe2/vm.generic-tensor::input-shape)
		    (cl-waffe2/vm.generic-tensor::tensor-input-shape tensor))
	      (setf (tensor-id use) (tensor-id tensor))
	      (setf (gethash key (vmalloc-id2pool allocation)) use)))
    allocation))

(defun free-allocate (allocation)
  "Deletes all tensors in memory-pool."
  (declare (type VMAllocation allocation))
  (loop for tensor being the hash-values in (vmalloc-id2pool allocation)
	for key    being the hash-keys   in (vmalloc-id2pool allocation) do
    (when (cl-waffe2/vm.generic-tensor::vec tensor)
      (funcall (tensor-finalizer tensor))
      (setf (gethash key (vmalloc-id2pool allocation)) nil))))

(defun storage-vec-from-memory-pool (allocation tensor)
  "Reading the allocated state, `allocation`, the function returns a storage vec of tensor."
  (declare (type VMAllocation allocation)
	   (type AbstractTensor tensor))

  (when (null (gethash (tensor-id tensor) (vmalloc-id2pool allocation)))
    (if (cl-waffe2/vm.generic-tensor::vec tensor)
	(return-from storage-vec-from-memory-pool (cl-waffe2/vm.generic-tensor::vec tensor))
	(error "tensor-vec: Attempted to read the storage vec of [~a, ~a, ~a] from memory-pool but failed because the tensor wasn't tracked when compiling.
Allocation State:
~a" (tensor-id tensor) (class-of tensor) (shape tensor) allocation)))
  
  (let ((result (gethash (tensor-id tensor) (vmalloc-id2pool allocation))))
    (when (null (cl-waffe2/vm.generic-tensor::vec result))
      (error "tensor-vec: In memory-pool, the InputTensor ~a isn't registered?" result))
    (setf (tensor-vec tensor) (cl-waffe2/vm.generic-tensor::vec result))
    (cl-waffe2/vm.generic-tensor::vec tensor)))

(defmacro with-static-allocation ((allocation) &body body)
  "
## [macro] with-static-allocation

Declares the static allocation state to use.
"
  `(let ((*static-alloc-state* ,allocation))
     (maybe-allocate! *static-alloc-state*)
     ,@body))

(defmacro with-protect-allocation (&body body)
  "Copies the current allocation"
  `(let ((*static-alloc-state* (copy-allocate *static-alloc-state*)))
     ,@body))

(defun assure-vmalloc ()
  (when (null *static-alloc-state*)
    (error "cl-waffe2 VM: forward/proceed seems to be executed without *static-alloc-state*. So the VM don't know what tensors to use.
Please explict the allocation state with: (with-static-allocation (allocation) ...)")))

(defun update-mempool-tensor (tensor value)
  (declare (type AbstractTensor tensor value))
  (assure-vmalloc)
  (setf (gethash (tensor-id tensor) (vmalloc-id2pool *static-alloc-state*)) value))

(defun read-from-mempool-tensor (tensor)
  (declare (type AbstractTensor tensor))
  (assure-vmalloc)
  (the AbstractTensor (or (gethash (tensor-id tensor) (vmalloc-id2pool *static-alloc-state*)) tensor)))


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

(defun inst-set-p (inst) (and (movetensor-p (wfop-node inst)) (movetensor-ignore-me (wfop-node inst))))

(defun eliminate-setq-node (iseq) ;; iseq[0] -> iseq[n]
  (let* ((setq-table (make-hash-table)))
    (loop for inst in iseq
	  if (inst-set-p inst)
	    do (setf (gethash (tensor-id (wfop-self inst)) setq-table) (tensor-id (second (wfop-args inst)))))

    (loop for inst in iseq
	  do (dolist (tensor `(,@(wfop-out-to inst) ,@(wfop-args inst)))
	       (let ((id (findout-origin setq-table tensor)))
		 (setf (tensor-id tensor) id))))

    (loop for inst in iseq
	  if (not (inst-set-p inst))
	    collect inst)))

(defun iseq-update-tensor-name! (iseq from to)
  (loop for inst in iseq do
    (dolist (o (wfop-out-to inst))
      (when (eql (tensor-id o) from)
	(setf (tensor-id o) to)))
    (dolist (a (wfop-args inst))
      (when (eql (tensor-id a) from)
	(setf (tensor-id a) to)))))
	
(defun optimize-memory-locality! (iseq-fw iseq-bw)
  (declare (type list iseq-fw)
	   (type (or null list) iseq-bw))

  ;; iseq-fw ... NIL is NG
  ;; iseq-bw ... NIL is ok

  (let* ((bw-leaves (make-hash-table))
	 (id2pool-table (make-hash-table))	 
	 (iseq `(,@(loop for inst in iseq-fw
			 if (null (wfop-block-iseq inst))
			   append (list inst)
			 else
			   append (wfop-block-iseq inst))))
	 (iseq-bw-flat (loop for inst in (reverse iseq-bw)
			     if (null (wfop-block-iseq inst))
			       append (list inst)
			     else
			       append (reverse (wfop-block-iseq inst)))))

    (when iseq-bw
      (loop for inst in iseq-bw-flat do
	(mapc #'(lambda (tensor)
		  (setf (gethash (tensor-id tensor) bw-leaves) tensor))
	      `(,@(wfop-out-to inst) ,@(wfop-args inst)))))
        
    (when iseq-bw-flat
      (apply-in-place-mutation! iseq-bw-flat (alexandria:hash-table-values bw-leaves))
      (setq iseq-bw-flat (reverse (eliminate-setq-node iseq-bw-flat))))

    (setq iseq `(,@iseq ,@(reverse iseq-bw-flat)))

    ;; Optimizes the locality of memory
    (simulate-memory-pool! iseq)

    ;; iseq ... flattened list of iseq
    ;; VM executes them in the order of iseq[0] iseq[1] ... iseq[n] where n = program_counter

    ;; Counting up all tensors (TMP Tensor) used in the node.
    ;; All we need is the first appearance in the node. So reverse it
    (loop for inst in (reverse `(,@iseq ,@iseq-bw-flat)) do
      (when (tensor-tmp-p (wfop-self inst))
	(setf (gethash (tensor-id (wfop-self inst)) id2pool-table) (wfop-self inst)))
      (mapc
       #'(lambda (arg)
	   (when (tensor-tmp-p arg)
	     (setf (gethash (tensor-id arg) id2pool-table) arg)))
       (wfop-args inst))
      
      (mapc
       #'(lambda (tensor)
	   (when tensor
	     (setf (gethash (tensor-id tensor) id2pool-table) tensor)))
       (wfop-sv4bw inst)))

;;    (maphash #'(lambda (x y)
;;		 (format t "~a->~a~%" x y))
;;	     id2pool-table)

    (values
     iseq-bw-flat
     (make-vmallocation
      :id2pool id2pool-table))))

(defun memory-pool-p (tensor1 tensor2)
  (and (equal (original-shape tensor1)
	      (original-shape tensor2))
       (eql (dtype tensor1) (dtype tensor2))))

(defun memory-pool-p< (tensor1 tensor2)
  (and (<= (apply #'* (original-shape tensor1))
	   (apply #'* (original-shape tensor2)))
       (eql (dtype tensor1) (dtype tensor2))))

(defun simulate-memory-pool! (iseq)
  (declare (optimize (speed 3))
	   (type list iseq))

  (%in-place-vm-ops! iseq)
  
  (let ((mempool-using-tensors nil)
	(pools nil)
	(pools-adj nil))
    (declare (type list pools pools-adj mempool-using-tensors))
    
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
		   (if (some #'symbolp (original-shape tensor))
		       (find tensor pools-adj :test #'memory-pool-p)
		       (find tensor
			     pools
			     :test #'memory-pool-p<))))
	     (read-from-pool (tensor)
	       (when (find (the symbol (tensor-id tensor)) mempool-using-tensors)
		 (return-from read-from-pool tensor))
	       
	       (if (tensor-tmp-p tensor)
		   (let ((out (find-from-pool tensor)))
		     ;; Delete from pool
		     (when (null out)
		       (push (tensor-id tensor) mempool-using-tensors)
		       (return-from read-from-pool tensor))
		     (push (tensor-id out) mempool-using-tensors)
		     (if (some #'symbolp (shape tensor))
			 (setq pools-adj (remove out pools-adj :test #'memory-pool-p :count 1))
			 (setq pools
			       (remove out pools :test #'memory-pool-p< :count 1)))
		     out)
		   tensor))
	     (set-as-free (tensor)
	       (when (and (tensor-tmp-p tensor)
			  ;; The tensor is from memory-pool?
			  (find (the symbol (tensor-id tensor)) mempool-using-tensors)
			  )
		 (setq mempool-using-tensors (delete (tensor-id tensor) mempool-using-tensors :test #'eql))
		 (if (some #'symbolp (original-shape tensor))
		     (push tensor pools-adj)
		     (push tensor pools)))))
      
      (loop for inst in iseq for pc fixnum upfrom 0 do
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
	  
	  (dolist (out out-to)
	    (let ((result (read-from-pool out)))
	      (when (not (eql (the symbol (tensor-id result)) (tensor-id out)))
		(iseq-update-tensor-name! (nthcdr pc iseq) (tensor-id out) (tensor-id result)))))
	  
	  (mapc #'(lambda (tensor state)
		    (when state
		      (set-as-free tensor)))
		args args-last-p))))))



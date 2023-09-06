
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

;; [TODO] Memory_Pool ... Tensor_IDベースでIndexを振り分ける
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
	 
	 (iseq `(,@(loop for inst in iseq-bw
			 if (null (wfop-block-iseq inst))
			   append (list inst)
			 else
			   append (wfop-block-iseq inst))
		 ,@(loop for inst in iseq-fw
			 if (null (wfop-block-iseq inst))
			   append (list inst)
			 else
			   append (wfop-block-iseq inst)))))

    ;; Counting up all tensors (TMP Tensor) used in the node.
    ;;(print iseq)
    (loop for inst in (reverse iseq) do ;; All we need is the first appearance in the node.
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
	     (print tensor)
	     (setf (gethash (tensor-id tensor) sv4bw-tensor-table) tensor)))
       (wfop-sv4bw inst)))

    ;; ~~~~~~~
    ;; ~~ 最適化 etc..
    ;; ~~~~~~~
    
    ;; [TODO] 前後でメモリ使用量計算して性能を評価する
    ;; [TODO] mempool-idxを入れ替えて最適化する
    
    (loop for tensor being the hash-values in cache-tensor-table do
      (setf (gethash (tensor-id tensor) alloc-route-table) (tensor-id tensor))
      (setf (gethash (tensor-id tensor) id->tensor-table) tensor))

    (loop for tensor being the hash-values in sv4bw-tensor-table do
      (setf (gethash (tensor-id tensor) alloc-route-table) (tensor-id tensor))
      (setf (gethash (tensor-id tensor) id->tensor-table) tensor))

    (print (make-vmallocation
	    :id->tensor id->tensor-table
	    :id-routes alloc-route-table))))

;;(tensor-mempool-idx tensor) を更新する・・・

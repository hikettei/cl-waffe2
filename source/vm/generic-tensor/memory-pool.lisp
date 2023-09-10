
(in-package :cl-waffe2/vm.generic-tensor)

;; [TODO] This file provides a global memory-pool

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;
;;  memory-pool.lisp provides features on MemoryPool, caching tensors, and managing allocation for dynamically shaped tensors.
;;
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; memory-pool.lisp is a thread-safe and cl-waffe2 dedicated memory-pool.

;; Tensors has a two state: ExistTensor and InputTensor
;;  As for ExistTensor, this file is not related anymore because their storage vec is stored in their own.
;;  However, memory-pool is important for InputTensor because their allocation is done lazily, and sometime changed
;;  And for efficiency, we have to manage their use status and cache.


;; - [ControlFlow of tensor-vec function] -------------------------------------------------------------------------
;;                  IF made by (make-tensor) ?
;;  [Tensor] ---> [ExistTensor] -> just returning (vec tensor) slot. (Shapes are static)
;;            |      made by (make-input `(10 10) nil) (<- IF name is set to nil, the InputTensor is recognised as CachedTensor.
;;            |--> [InputTensor] -> We use memory-pool, because the number of A and B could be changed later.
;;            |      made by (make-input `(A B) :X)
;;            |--> [InputTensor] -> just returning (vec tensor) slot is enough. (allocated by set-input)
;;            |
;;           ...
;; ----------------------------------------------------------------------------------------------------------------


;; Dealing with dynamycally shaped tensors:
;;   In cl-waffe2, the size of InputTensor dynamically changes depeneding on batch-size.
;;   That is, X=(BATCH 100) tensor also could be: (1 100) or (100 100) tensor.
;;   If the total size of tensor resized, is larger than before allocated one, do a reallocate otherwise use old one.
;;  
;;  Case1. BATCH=1 -> BATCH=100
;;   => Reallocate or find 1 100 Tensor
;;
;;  Case2. BATCH=100 -> BATCH=1
;;   => Return (100 100) Tensor as (1 100) Tensor's strides.
;;

;; Keep Thread-Safe:
;;  Memory-Pool is devided to each thread managed by bordeaux-threads to avoid conflicts.
;;  The toplevel variable is a *thread-memory-pool*, and branching like:
;;   *thread-memory-pool*
;;        thread-idx
;;            1       -> [struct] Memory-Pool
;;            2       -> [struct] Memory-Pool
;;            3       -> [struct] Memory-Pool
;;                    ...


;; Caching Tensors:
;;  First, tensors in the memory-pool has a these state:
;;   InputTensor[state=:input]
;;      - Whenever the computation done and the allocation isn't needed anymore, ...
;;        Tensors with state=:input would never moved into cached-room
;;        InputTensor with their name is a keyword, is register as :input
;;   InputTensor[state=:tmp]
;;      - InputTensors created like (make-input `(...) nil) is registered as :tmp
;;      - The storage vec isn't guaranteed to be filled with 0.
;;      - When ref-count become 0, they're moved into cached-tensors
;;      - 
;;   InputTensor[state=:save-for-backward]
;;      - When *no-grad*=nil, becomes :input
;;      - When *no-grad*=t,   :tmp


(declaim (inline get-from-global-memory-pool))

(defparameter *thread-memory-pool*
  (make-hash-table)
  "A weak-hash-table which restores: [Thread-IDX] -> [Memory-Pool]")

(defvar *thread-pool-lock* (make-lock "thread cache lock"))

;; It was originally (defvar *adjustable-shape-table* nil)
;; If there's any shape-error related to static composite functions, doubt this:
(defparameter *adjustable-shape-table* (make-hash-table) "An hash-table: Symbol -> Size.") ;; (A -> 10, B -> 10)

(defvar *current-shape-state* nil) ;; Global Adjustable-Shape-State

;; [TODO] No Runtime Allocation
(deftype mem-pool-state-t ()
  "
:save-for-backward ...  Room for save4backward
:tmp               ...  When quitting with-mem-pool, the value of :tmp is freed or reused.
:input             ...  When quitting with-mem-pool, the room is extended to the superior scope.
"
  `(or null (and keyword (member :save-for-backward :tmp :input))))

(deftype read-io-state-t ()
  `(or null (and keyword (member :using :free))))

(defstruct Memory-Pool
  "
When TMP Tensor (created by (make-input `(...) nil) function) is required to access its storage vector:
    1. Try to finds out freed storage vec(where io=:free) from cached-pool of thread.
    2. If found, use this, otherwise allocate the new one (set io=:using).

In order to set the io of each cache-pool as :free, cl-waffe2/vm traces the cl-waffe2 IR and notes the last reference.
"
  (temporary-rooms (make-hash-table) :type hash-table) ;; All Tensors allocated here
  (cached-pool     (make-hash-table) :type hash-table) ;; Tensors with :free :using states. (apply #'* (translate-adjustable-shape (slot-value 'orig-shape))) -> ((room TENSOR1) (room TENSOR2) (room TENSOR3) ...)
  )

(defun current-memory-pool (&optional (idx nil))
  "Set idx to force the access of idxth thread."
  (let ((memory-pool (with-lock-held (*thread-pool-lock*)
		       (gethash (or idx (current-thread)) *thread-memory-pool*))))
    (if memory-pool
	memory-pool
	(with-lock-held (*thread-pool-lock*)
	  (setf (gethash (or idx (current-thread)) *thread-memory-pool*) (make-memory-pool))))))


;; [TODO] Add: :gradient :cache :input
;;        Add: if the facet=:cache, reuse

(defstruct (Temporary-Room
	    (:constructor make-room
		(input-tensor
		 &aux
		   (shape-first (shape input-tensor)))))
  ;; Adjustable Tensor Size
  (state nil :type mem-pool-state-t)
  (shape-first shape-first :type list)
  (io-state :using :type read-io-state-t)
  (size (apply #'* (translate-adjustable-shape (shape input-tensor))) :type fixnum)
  (cache-tensor input-tensor :type AbstractTensor))

(defun print-current-memory-pool (&key (stream t))
  (print (current-memory-pool) stream)
  nil)

(defmethod print-object ((pool Memory-Pool) stream)
  (let ((exist-tensors)
	(non-exist-tensors)
	(print-elements1)
	(print-elements2))
    (maphash #'(lambda (key value &aux (tensor (temporary-room-cache-tensor value)))
		 (if (vec tensor)
		     (push (cons key value) exist-tensors)
		     (push (cons key value) non-exist-tensors)))
	     (memory-pool-temporary-rooms pool))

    (dolist (tensor exist-tensors)
      (push (format nil "[~a -> ~a]" (car tensor) (temporary-room-shape-first (cdr tensor))) print-elements1)
      (push " | " print-elements1))

    (dolist (tensor non-exist-tensors)
      (push (format nil "[~a -> ~a]" (car tensor) (temporary-room-shape-first (cdr tensor))) print-elements2)
      (push " | " print-elements2))

    (format stream "
+= [Memory Pool] =======================+
")
    
    (mapc
     #'(lambda (print-elements display-name)
	 (format stream display-name)
	 (format stream "
~a"
		 (with-output-to-string (out)
		   (dolist (e print-elements) (princ e out)))))
     `(,print-elements1 ,print-elements2)
     `(" + [Allocated] +" "~% + [Unallocated] +"))
    nil))

(defun free-current-memory-pool ()
  ""
  ;; TODO:
  ;; (maphash ... tensor-delete)
  ;; Maybe:: CUDA Foreign Pointers aren't gc-reachable??
  ;;(exit-memory-pool)
  (setf *thread-memory-pool* (make-hash-table));;(tg:make-weak-hash-table :weakness :key))
  #+sbcl(sb-ext:gc :full t)
  )

;; Users do not have to use this macro.
(defmacro with-memory-pool (&body body &aux (result (gensym)))
  "
## [macro] with-memory-pool

InputTensors created inside this macro guarantees for all read information to be :free inside it.

"
  `(let* ((,result
	    (let ((*thread-memory-pool* (make-hash-table)))
	      (multiple-value-list
	       (progn ,@body)))))
     (apply #'values ,result)))

(defun get-mem-pool (key)
  (gethash key (memory-pool-temporary-rooms (current-memory-pool))))

(defun set-mem-pool (key value &optional (idx nil))
  (setf (gethash key (memory-pool-temporary-rooms (current-memory-pool idx))) value)
  value)

(defun alloc-from-input-tensor (tensor)
  (declare (type AbstractTensor tensor))
  (let ((out (make-tensor (original-shape tensor)
			  :requires-grad nil
			  :dtype (dtype tensor)
			  :order (order tensor)
			  :device (class-of tensor))))
    (setf (tensor-vec tensor) (vec out))
    out))

(defun chaintmp-find-mem-pool (tensor)
  (declare (type AbstractTensor)
	   (optimize (speed 3)))
  
  ;; Assert: The Given Tensor is ChaimTMP
  ;; Set current-state for future allocating
  
  (let ((place (tensor-id tensor)))
    (let ((room (get-mem-pool place)))
      (if room
	  (vec (temporary-room-cache-tensor room))
	  (vec (temporary-room-cache-tensor
		(set-mem-pool place (make-room (alloc-from-input-tensor tensor)))))))))

(defun get-from-global-memory-pool (tensor)
  (declare (type AbstractTensor tensor)
	   (optimize (speed 3)))
  
  (when (some #'symbolp (shape tensor))
    (error "tensor-vec: Can't allocate the InputTensor on memory because the given one still inludes a symbol.
~a

-> Compile the tensor with `build`, and set inputs." tensor))

  (cond
    ((scalar-p tensor) ;; As of ScalarTensor, memory-usage = O(1)
     (if (vec tensor)
	 (vec tensor)
	 (let ((tmp-tensor (make-tensor 0 :dtype (dtype tensor) :order (order tensor))))
	   (setf (tensor-vec tensor) (vec tmp-tensor))
	   (vec tensor))))
    ;; If storage is nil, allocate anyway
    (T (chaintmp-find-mem-pool tensor))))


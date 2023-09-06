
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

(declaim (inline get-from-memory-pool))
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

(defstruct Adjustable-Shape-State
  ;; An copy of *adjustable-shape-table* to detect shape changing later.
  (state (alexandria:copy-hash-table *adjustable-shape-table*) :type hash-table))

(defun adjustable-shape-compatible (old-state)
  (declare (type Adjustable-Shape-state old-state)
	   (optimize (speed 3)))
  (let ((current-keys (loop for symbol being the hash-keys   in *adjustable-shape-table* collect symbol))
	(current-vals (loop for size   being the hash-values in *adjustable-shape-table* collect size))
	(old-table    (adjustable-shape-state-state old-state)))
    (not (some #'(lambda (target-symbol current-val)
		   (let ((val (gethash target-symbol old-table)))
		     (if (and (typep val 'fixnum)
			      (typep current-val 'fixnum))
			 ;; Ex: The tensor is created as BATCH-SIZE=100, and now the operation is going udner BATCH_SIZE=10...
			 (< val current-val)
			 t)))
	       current-keys current-vals))))

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
  (exit-memory-pool)
  ;;(setf *thread-memory-pool* (make-hash-table));;(tg:make-weak-hash-table :weakness :key))
  #+sbcl(sb-ext:gc :full t)
  )

(defun exit-memory-pool ()
  "Return: An list of AbstractTensor which wants to be registered to the superior scope"
  (let ((extended))
    (maphash
     #'(lambda (thread-idx thread-memory-pool)
	 (print thread-idx)
	 (maphash
	  #'(lambda (key room)
	      (print key)
	      (let ((state (temporary-room-state room))
		    (tensor (temporary-room-cache-tensor room)))
		;; state = :tmp :input :save-for-backward
		(cond
		  ((and (not *no-grad*) (eql state :save-for-backward))
		   (funcall (tensor-finalizer tensor)))
		  ((or (null state) (eql state :tmp))
		   (funcall (tensor-finalizer tensor)))
		  (T
		   (push (list key room thread-idx) extended)))))
	  (memory-pool-temporary-rooms thread-memory-pool)))
     *thread-memory-pool*)
    (setf *thread-memory-pool* (make-hash-table))
    extended))

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
     (dolist (i (exit-memory-pool))
       (set-mem-pool (symbol-name (first i)) (second i) (third i)))
     (apply #'values ,result)))

(defun get-mem-pool (key)
  (declare (type string key))
  (gethash (intern key "KEYWORD") (memory-pool-temporary-rooms (current-memory-pool))))

(defun set-mem-pool (key value &optional (idx nil))
  (declare (type string key))
  (setf (gethash (intern key "KEYWORD") (memory-pool-temporary-rooms (current-memory-pool idx))) value)
  value)

(defun assure-and-return-room (room tensor)
  "Checking room's size, returning tensor"
  (declare (type Temporary-Room room)
	   (type AbstractTensor tensor)
	   (optimize (speed 3)))
  (let ((required-size (apply #'* (translate-adjustable-shape (original-shape tensor))))
	(vec           (vec       (temporary-room-cache-tensor room))))

    ;; Checking required-size, is done at toplevel.
    ;; Use (max-size) x (max-size) vec as if they're (required-size) x (required-size) vec.

    ;; TODO: Add a new attribute Room: :read-state to reuse memory-pool which is not used.
    ;; :read-state is one of: :used :free-now
    
    ;; when assure-and-return-room is called:
    ;; find :free-now and required-size is enough caches, and return it.

    ;; Each time update room, the operation works correctly???

    (when (or (null vec) (null (vec tensor))
	      (not (eql (the keyword (dtype tensor)) (the keyword (dtype (temporary-room-cache-tensor room)))))
	      (> (the fixnum required-size) (temporary-room-size room)))
      ;; Update memory-pool
      (setf (tensor-alloc-state tensor) *current-shape-state*)
      (setf (temporary-room-size room) required-size)
      (setf (tensor-vec (temporary-room-cache-tensor room))
	    (vec (make-tensor `(,required-size) :dtype (dtype tensor) :order (order tensor))))

      (setf (tensor-vec tensor) (vec (temporary-room-cache-tensor room))))
    (vec tensor)))

(defun chaintmp-find-mem-pool (tensor)
  (declare (type AbstractTensor)
	   (optimize (speed 3)))
  
  ;; Assert: The Given Tensor is ChaimTMP
  ;; Set current-state for future allocating
  
  (let ((place (tensor-name tensor)))
    (declare (type string place))
    (let ((room (get-mem-pool place)))
      (if room
	  (assure-and-return-room room tensor)
	  (and (set-mem-pool place (make-room tensor)) ;; set and read it again
	       (chaintmp-find-mem-pool tensor))))))


(defun get-from-memory-pool (tensor)
  (declare (type AbstractTensor tensor)
	   (optimize (speed 3)))

  (when (some (the function (compose #'symbolp #'read-symbol)) (shape tensor))
    (error "tensor-vec: Can't allocate the new space for the given InputTensor:
~a
because its shape still includes a symbol." tensor))
  
  (cond
    ((scalar-p tensor) ;; As of ScalarTensor, memory-usage = O(1)
     (if (vec tensor)
	 (vec tensor)
	 (let ((tmp-tensor (make-tensor 0 :dtype (dtype tensor) :order (order tensor))))
	   (setf (tensor-vec tensor) (vec tmp-tensor))
	   (vec tensor))))
    ;; If storage is nil, allocate anyway
    ((null (vec tensor))
     (chaintmp-find-mem-pool tensor))
    
    ((null (tensor-alloc-state tensor))
     ;; First Time Allcoation? or no one uses adjustable-symbol?
     (vec tensor))

    ;; If the tensor's size is STATIC -> just returning vec with finding mem-pool
    ((every #'numberp (tensor-input-shape tensor))
     (vec tensor))
    
    ;; Other case, the shape of tensors has a symbol, but keep using the same one.
    ((no-need-update-p tensor)
     (vec tensor))
    ;; Otherwise, reallocating/resizing is needed!
    ((stringp (tensor-name tensor))
     (chaintmp-find-mem-pool tensor))
    ;; The Tensor is InputTensor (ChainTMP)
    ((user-input-p tensor) ;; user-input-p ... expensive
     (error "get-from-memory-pool failed: ~a isn't embodied." tensor))
    (T
     (error "get-from-memory-pool failed: because the given tensor isn't one of: ScalarTensor InputTensor(ChainTMP)"))))

(defun write-mempool-state (tensor attribute)
  (declare (type AbstractTensor tensor)
	   (type mem-pool-state-t attribute))

  (let ((room (get-mem-pool (tensor-name tensor))))
    (when room ;; When room=nil, there's no need to consider whether the tensor will be freed or not.
      (setf (temporary-room-state room) attribute)))
  nil)


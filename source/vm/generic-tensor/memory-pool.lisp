
(in-package :cl-waffe2/vm.generic-tensor)

;;
;; memory-pool.lisp needs to be given more thoughts.
;;
;;
;; memory-pool.lisp is a thread-safe and generic memory-pool program.
;; ExistTensor is not related to this file because it stores a vec for each tensor.
;; The topic is for InputTensor: (cache Tensor, the result of operations, sometimes arguments)
;;

;; = [Flow of (tensor-vec) ] ====================
;;                 made by (make-tensor) ?
;; [Tensor] ---> [ExistTensor] -> just returning (vec tensor) slot. (Shapes are static)
;;          |      made by (make-input `(10 10) nil) (<- Setting nil = the tensor is cache)
;;          |--> [InputTensor] -> use memory-pool, because it is just temporary, and shapes are dynamically changing
;;          |      made by (make-input `(10 10) :X)
;;          |--> [InputTensor] -> just returning (vec tensor) slot is enough. (allocated by set-input)
;;          |
;;

;; = [1. Deal with Adjustable Shape] ==================================================
;; In cl-waffe2, the size of InputTensor dynamically changes depeneding on batch-size.
;; That is, X=(BATCH 100) tensor could be: (1 100) or (100 100) tensor.
;; 
;; Case1. BATCH=1 -> BATCH=100
;; => Reallocate or find 1 100 Tensor
;;
;; Case2. BATCH=100 -> BATCH=1
;; => Return (100 100) Tensor as (1 100) Tensor's strides.
;;
;; = [2. Reusing cache] ==================================================
;; POOL = [[ROOM (100 100) :USING] [ROOM :FREE] [ROOM :USING] ...]
;; Each room has two state: :USING AND :FREE (TODO)
;; 
;;
;; = [3. Thread Safe]   ==================================================
;; Memory-Pool is created for each Threads.
;;
;; [Table] *thread-memory-pool*
;; Thread1 => POOL1
;; Thread2 => POOL2
;;        ...
;;


(defparameter *thread-memory-pool*
  (make-hash-table);;(tg:make-weak-hash-table :weakness :key)
  "A weak-hash-table which restores: [Thread-IDX] -> [Memory-Pool]")
(defvar       *thread-pool-lock*   (make-lock "thread cache lock"))

(defvar *adjustable-shape-table* nil "An hash-table: Symbol -> Size.") ;; (A -> 10, B -> 10)
(defvar *current-shape-state*    nil) ;; Global Adjustable-Shape-State

(defstruct Memory-Pool
  (temporary-rooms (make-hash-table) :type hash-table))

(defstruct Adjustable-Shape-State
  ;; An copy of *adjustable-shape-table* to detect shape changing later.
  (state (alexandria:copy-hash-table *adjustable-shape-table*) :type hash-table))

;; Could be slightly slow...
(defun adjustable-shape-compatible (old-state)
  (declare (type Adjustable-Shape-state old-state)
	   (optimize (speed 3)))
  (let ((current-keys (loop for symbol being the hash-keys   in *adjustable-shape-table* collect symbol))
	(current-vals (loop for size   being the hash-values in *adjustable-shape-table* collect size))
	(old-table    (adjustable-shape-state-state old-state)))
    ;; (not some
    (not (some #'(lambda (target-symbol current-val)
		   (let ((val (gethash target-symbol old-table)))
		     (if (and (typep val 'fixnum)
			      (typep current-val 'fixnum))
		       ;; The tensor is created as BATCH-SIZE=100, Now the operation is going udner BATCH_SIZE=10			  

			 (< val current-val)
			 t)))
	       current-keys current-vals))))

(defun current-memory-pool ()
  (let ((memory-pool (with-lock-held (*thread-pool-lock*)
		       (gethash (current-thread) *thread-memory-pool*))))
    (if memory-pool
	memory-pool
	(with-lock-held (*thread-pool-lock*)
	  (setf (gethash (current-thread) *thread-memory-pool*) (make-memory-pool))))))

(defstruct (Temporary-Room
	    (:constructor make-room
		(input-tensor
		 &aux
		   (shape-first (shape input-tensor)))))
  ;; Adjustable Tensor Size
  (shape-first shape-first :type list)
  (size (apply #'* (translate-adjustable-shape (shape input-tensor))) :type fixnum)
  (cache-tensor input-tensor :type AbstractTensor))

(defun print-current-memory-pool (&key (stream t))
  (let ((pool (current-memory-pool)))
    (print pool stream)
    nil))

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
  ;; TODO:
  ;; (maphash ... tensor-delete)
  ;; Maybe:: CUDA Foreign Pointers aren't gc-reachable??
  (setf *thread-memory-pool* (make-hash-table));;(tg:make-weak-hash-table :weakness :key))
  #+sbcl(sb-ext:gc :full t)
  )

(defmacro with-memory-pool (&body body)
  "
## [macro] with-memory-pool

Creates a new scope of memory-pool.

After the body exists, all the temporary tensors in the pool is freed.
"
  `(let ((*thread-memory-pool* (make-hash-table)));;(tg:make-weak-hash-table :weakness :key)))
     (unwind-protect (progn ,@body)
       (free-current-memory-pool))))

;; To USE :EQL instead of equal, String->Keyword
(defun get-mem-pool (key)
  (declare (type string key))
  (gethash (intern key "KEYWORD") (memory-pool-temporary-rooms (current-memory-pool))))

(defun set-mem-pool (key value)
  (declare (type string key))  
  (setf (gethash (intern key "KEYWORD") (memory-pool-temporary-rooms (current-memory-pool))) value)
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

(defun translate-adjustable-shape (shape) ;; tensor-input-shape
  "
## [function] translate-adjustable-shape

Reading the *adjustable-shape-table*, the function returns an list consisted of fixnum.

If there's any undetermined one, returns an error (TODO: Add Conditions)"
  (declare (type list shape)
	   (optimize (speed 3)))
  (map 'list #'read-symbol shape))

(defun read-adjustable-symbol (s)
  (typecase s
    (fixnum s)
    (symbol
     (or (read-symbol s);;(gethash s *adjustable-shape-table*)
	 (error "translate-adjustable-shape: encountered unknown symbol: ~a" s)))))

(defmacro with-adjustable-symbol ((symbol-name symbol-value) &body body)
  "Adding an element: symbol-name -> symbol-value to *adjustable-shape-table*, which can be read by translate-adjustable-shape function.

Usage:

(with-adjustable-symbols (('a 1) ('b 1))
    (with-let-adjustable-symbols (a b)
        (print a)
        (print b)))

"

  `(let* ((*adjustable-shape-table* (or *adjustable-shape-table* (make-hash-table)))
	  (*current-shape-state*    (make-adjustable-shape-state)))
     
     ;;(unless (typep ,symbol-value 'fixnum)
     ;;  (error "with-adjustalbe-symbol: Attempted to register an symbol, ~a (TODO More clear error)" ,symbol-value))
     
     (setf (gethash ,symbol-name *adjustable-shape-table*) ,symbol-value)
     ,@body))

(defmacro with-adjustable-symbols ((&rest forms) &body body)
  (labels ((expand-form (rest-forms)
	     (if (null rest-forms)
		 `(progn ,@body)
		 `(with-adjustable-symbol (,@(car rest-forms))
		    ,(expand-form (cdr rest-forms))))))
    (expand-form forms)))

(defmacro with-let-adjustable-symbol (symbol-name &body body)
  `(let ((,symbol-name (gethash ',symbol-name *adjustable-shape-table*)))
     (declare (type fixnum ,symbol-name)
	      
	      (ignorable ,symbol-name))
     (declare (ignore symbol-name))
     ,@body))

;; Fix: symbol conflicts??
(defmacro with-let-adjustable-symbols ((&rest symbol-names) &body body)
  (labels ((expand (rest-forms)
	     (if (null rest-forms)
		 `(progn ,@body)
		 `(with-let-adjustable-symbol ,(car rest-forms)
		    ,(expand (cdr rest-forms))))))
    (expand symbol-names)))

(defun read-symbol (symbol)
  (if *adjustable-shape-table*
      (typecase symbol
	(symbol
	 ;; out <- table[symbol]
	 (let ((out (gethash symbol *adjustable-shape-table*)))
	   (if (null out)
	       symbol ;; No result?
	       (if (symbolp out)
		   ;; A = BATCH_SIZE = 10
		   (read-symbol out)
		   out))))
	;; Return fixnum
	(T symbol))
      symbol))

(defun no-need-update-p (tensor)
  (declare (type AbstractTensor tensor))
  (adjustable-shape-compatible (tensor-alloc-state tensor)))

(defun get-from-memory-pool (tensor)
  (declare (type AbstractTensor tensor)
	   (optimize (speed 3)))

  ;; The number of call cases:

  ;;  Much Higher  <->    Low
  ;;    ChainTMP        ScalarTensor
  (cond
    ((scalar-p tensor)
     (if (vec tensor)
	 (vec tensor)
	 (let ((tmp-tensor (make-tensor 0 :dtype (dtype tensor) :order (order tensor))))
	   (setf (tensor-vec tensor) (vec tmp-tensor))
	   (vec tensor))))
    ((null (tensor-alloc-state tensor))
     ;; First Time Allcoation? or no one uses adjustable-symbol?
     (or (vec tensor)
	 (chaintmp-find-mem-pool tensor)))
    ;; If the tensor's size is STATIC -> just returning vec with finding mem-pool
    ((or (null (tensor-input-shape tensor))
	 (every #'numberp (tensor-input-shape tensor)))
     (if (vec tensor)
	 (vec tensor)
	 (chaintmp-find-mem-pool tensor)))
    ;; Other case, the shape of tensors has a symbol, but keep using the same one.
    ((no-need-update-p tensor)
     (if (vec tensor)
	 (vec tensor)
	 (chaintmp-find-mem-pool tensor)))
    ;; Otherwise, reallocating needs to be done.
    ((stringp (tensor-name tensor))
     (chaintmp-find-mem-pool tensor))
    ;; The Tensor is InputTensor (ChainTMP)
    ;; (user-input-p tensor) is expensible, so use stringp instead.
    ((user-input-p tensor) ;; high cost
     (error "get-from-memory-pool failed: ~a isn't embodied." tensor))
    (T
     (error "get-from-memory-pool failed: because the given tensor isn't one of: ScalarTensor InputTensor(ChainTMP)"))))


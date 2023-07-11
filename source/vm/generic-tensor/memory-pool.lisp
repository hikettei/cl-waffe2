
(in-package :cl-waffe2/vm.generic-tensor)

;;
;; memory-pool.lisp is an file to manage temporary tensors with adjustable-symbols.
;;

(defstruct Memory-Pool
  ;; Gc-able temporary-rooms?
  (temporary-rooms (make-hash-table) :type hash-table))

;; TODO: AddState :using :freenow
(defparameter *memory-pool* (make-memory-pool) "Memory-Pool is a place to store caching tensors.")

(defvar *adjustable-shape-table* nil "An hash-table: Symbol -> Size.")

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
  (print-object *memory-pool* stream))


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
  ;; TODO
  ;; *memory-pool*
  ;; (maphash ... tensor-delete)
  (setf *memory-pool* (make-memory-pool))
  )

(defmacro with-memory-pool (&body body)
  "
## [macro] with-memory-pool

Creates a new scope of memory-pool.

After the body exists, all the temporary tensors in the pool is freed."
  `(let ((*memory-pool* (make-memory-pool)))
     (unwind-protect (progn ,@body)
       (free-current-memory-pool))))

;; To avoid using equal which is heavy operation, coerce string -> keyword:
(defun get-mem-pool (key)
  (declare (type string key))
  (gethash (intern key "KEYWORD") (memory-pool-temporary-rooms *memory-pool*)))

(defun set-mem-pool (key value)
  (declare (type string key))
  ;;(tg:finalize value #'(lambda () (remhash key (memory-pool-temporary-rooms *memory-pool*))))
  
  (setf (gethash (intern key "KEYWORD") (memory-pool-temporary-rooms *memory-pool*)) value)
  value)

(defun assure-and-return-room (room tensor)
  "Checking room's size, returning tensor"
  (declare (type Temporary-Room room)
	   (type AbstractTensor tensor)
	   (optimize (speed 3)))
  (let ((required-size (apply #'* (translate-adjustable-shape (original-shape tensor))))
	(vec           (vec (temporary-room-cache-tensor room))))

    ;; Checking required-size, is done at toplevel.
    ;; Use (max-size) x (max-size) vec as if they're (required-size) x (required-size) vec.

    ;; TODO: Add a new attribute Room: :read-state
    ;; :read-state is one of: :used :free-now
    ;; when assure-and-return-room is called:
    ;; find :free-now and required-size is enough caches, and return it.

    ;; Each time update room, the operation works correctly???

    (when (or (null vec)
	      (> (the fixnum required-size) (temporary-room-size room)))
      (setf (temporary-room-size room) required-size)
      (setf (tensor-vec (temporary-room-cache-tensor room))
	    (vec (make-tensor `(,required-size) :dtype (dtype tensor) :order (order tensor)))))

    (when (null (vec tensor))
      (setf (tensor-vec tensor) (vec (temporary-room-cache-tensor room))))

    (vec tensor)))

(defun chaintmp-find-mem-pool (tensor)
  (declare (type AbstractTensor)
	   (optimize (speed 3)))

  ;; Assert: The Given Tensor is ChaimTMP

  (let ((place (tensor-name tensor)))
    (declare (type string place))

    (let ((room (get-mem-pool place)))
      (if room
	  (assure-and-return-room room tensor)
	  (and (set-mem-pool place (make-room tensor)) ;; set and read it
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

  `(let ((*adjustable-shape-table* (or *adjustable-shape-table* (make-hash-table))))
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

(defun get-from-memory-pool (tensor)
  (declare (type AbstractTensor tensor)
	   (optimize (speed 3)))

  ;; The number of call cases:

  ;;  Much Higher  <->    Low
  ;;    ChainTMP        ScalarTensor
  (cond
    ;; The Tensor is Scalar
    ((scalar-p tensor)
     (let ((tmp-tensor (make-tensor 0 :dtype (dtype tensor) :order (order tensor))))
       (setf (tensor-vec tensor) (vec tmp-tensor))
       (vec tensor)))
    ((stringp (tensor-name tensor))
     (chaintmp-find-mem-pool tensor))
    ;; The Tensor is InputTensor (ChainTMP)
    ;; (user-input-p tensor) is expensible, so use stringp instead.
    ((user-input-p tensor) ;; high cost
     (error "get-from-memory-pool failed: ~a isn't embodied." tensor))
    (T
     (error "get-from-memory-pool failed: because the given tensor isn't one of: ScalarTensor InputTensor(ChainTMP)"))))


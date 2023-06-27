
(in-package :cl-waffe2/vm.generic-tensor)

;;
;; memory-pool.lisp is an file to manage temporary tensors with adjustable-symbols.
;;

(defstruct Memory-Pool
  (temporary-rooms nil :type list))

(defun free-current-memory-pool ()
  
  )

(defparameter *memory-pool* (make-memory-pool) "Memory-Pool is a place to store caching tensors.")

(defvar *adjustable-shape-table* nil "An hash-table: Symbol -> Size.")


(defstruct (Temporary-Room
	    (:constructor make-room (input-tensor)))
  ;; Adjustable Tensor Size
  (cache-tensor input-tensor :type AbstractTensor))

(defmacro with-memory-pool (&body body)
  "
## [macro] with-memory-pool

Creates a new scope of memory-pool.

After the body exists, all the temporary tensors in the pool is freed."
  `(let ((*memory-pool* (make-memory-pool)))
     (unwind-protect (progn ,@body)
       (free-current-memory-pool))))

(defun translate-adjustable-shape (shape) ;; tensor-input-shape
  "
## [function] translate-adjustable-shape

Reading the *adjustable-shape-table*, the function returns an list consisted of fixnum.

If there's any undetermined one, returns an error (TODO: Add Conditions)"
  (declare (type list shape)
	   (optimize (speed 3)))
  (loop for s in shape
	collect (typecase s
		  (fixnum s)
		  (symbol
		   (or (gethash s *adjustable-shape-table*)
		       (error "translate-adjustable-shape: encountered unknown symbol: ~a" s))))))

(defmacro with-adjustable-symbol ((symbol-name symbol-value) &body body)
  "Adding an element: symbol-name -> symbol-value to *adjustable-shape-table*, which can be read by translate-adjustable-shape function.

Usage:

(with-adjustable-symbols (('a 1) ('b 1))
    (with-let-adjustable-symbols (a b)
        (print a)
        (print b)))

"

  `(let ((*adjustable-shape-table* (or *adjustable-shape-table* (make-hash-table))))
     (unless (typep ,symbol-value 'fixnum)
       (error "with-adjustalbe-symbol: Attempted to register an symbol, ~a (TODO More clear error)" ,symbol-value))
     
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
     ,@body))

(defmacro with-let-adjustable-symbols ((&rest symbol-names) &body body)
  (labels ((expand (rest-forms)
	     (if (null rest-forms)
		 `(progn ,@body)
		 `(with-let-adjustable-symbol ,(car rest-forms)
		    ,(expand (cdr rest-forms))))))
    (expand symbol-names)))

(defun read-symbol (symbol)
  (gethash symbol *adjustable-shape-table*))


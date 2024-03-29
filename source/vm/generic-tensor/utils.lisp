
(in-package :cl-waffe2/vm.generic-tensor)

(defun map-tree (fn tree)
  (let ((tree (funcall fn tree)))
    (if (listp tree)
        (mapcar (lambda (subtree)
                  (map-tree fn subtree))
                tree)
        tree)))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

(define-modify-macro multf (&optional (number 1)) *)

(defun compose (&rest fns)
  (if fns
      (let ((fn1 (car (last fns)))
            (fns (butlast fns)))
        #'(lambda (&rest args)
                   (reduce #'funcall fns
                           :from-end t
                           :initial-value (apply fn1 args))))
      #'identity))

(defun wrap-x (x)
  (typecase x
    (number `(the fixnum ,x))
    (symbol `(the fixnum (cl-waffe2/vm:maybe-observe-axis ',x)))
    (T `(the fixnum ,x))))

(defun lazy* (x y)
  (if (and (numberp x)
	   (numberp y))
      (* x y)
      `(the fixnum
	    (* ,(wrap-x x)
	       ,(wrap-x y)))))

(defun lazy-mulup (&rest args)
  (let ((res 1))
    (dolist (arg args) (setq res (lazy* res arg)))
    res))

(declaim (ftype (function (list) list)
		column-major-calc-strides
		row-major-calc-strides))
(defun column-major-calc-strides (shape)
  (declare (type list shape))
  (let* ((num-dims (length shape))
         (strides (make-list num-dims :initial-element 1)))
    (loop for i downfrom (- num-dims 2) to 0 do
      (setf (nth i strides) (lazy* (nth (+ i 1) strides)
				   (nth (+ i 1) shape))))
    strides))


(defun row-major-calc-strides (shape)
  (declare (type list shape))
  (let* ((num-dims (length shape))
         (strides (make-list num-dims :initial-element 1)))
    (loop for i from 1 to (- num-dims 1) do
      (setf (nth i strides) (lazy* (nth (- i 1) strides)
				   (nth (- i 1) shape))))
    strides))

(defmacro let*-ignorable ((&rest forms) &body body)
  (labels ((expand-forms (rest-forms)
	     (if rest-forms
	       `(let (,(car rest-forms))
		  (declare (ignorable ,(caar rest-forms)))
		  ,(expand-forms (cdr rest-forms)))
	       `(progn ,@body))))
    (expand-forms forms)))

(defun use-number-one (a b)
  (if (numberp a)
      a
      b))

(defun user-input-p (tensor)
  (and (eql (tensor-facet tensor) :input)
       (eql (tensor-attribute tensor) :input)))


;; [TODO] Extend devices
(defun make-clone (tensor &optional name ignore-create-from)
  (let* ((shape (actual-shape tensor))
	 (out (make-input shape (or name nil)
			  :create-from (if ignore-create-from
					   nil
					   tensor)
			  :dtype (dtype tensor)
			  :order (order tensor)
			  :scalar-p (scalar-p tensor)))
	 (broadcasted-p)
	 (broadcasts (loop for size in (shape tensor)
			   for view in (tensor-view tensor)
			   if (eql :broadcast (viewtype (force-list view)))
			     collect (and
				      (setq broadcasted-p t)
				      `(:broadcast ,size))
			   else
			     collect t))
	 (out (if broadcasted-p
		  (apply #'view out broadcasts)
		  out)))
    
    (setf (tensor-initial-offset out) (tensor-initial-offset tensor))
    out))

(defun make-clone-exist (tensor)
  (if (scalar-p tensor)
      (make-tensor 0 :device (class-of tensor) :dtype (dtype tensor) :order (order tensor))
      (let* ((shape (actual-shape tensor))
	     (out (make-tensor shape
			       :create-from tensor
			       :device (class-of tensor)
			       :dtype (dtype tensor)
			       :order (order tensor)))
	     (broadcasted-p)
	     (broadcasts (loop for size in (shape tensor)
			       for view in (tensor-view tensor)
			       if (eql :broadcast (viewtype (force-list view)))
				 collect (and
					  (setq broadcasted-p t)
					  `(:broadcast ,size))
			       else
				 collect t))
	     (out (if broadcasted-p
		      (apply #'view out broadcasts)
		      out)))
	(setf (tensor-initial-offset out) (tensor-initial-offset tensor))
	out)))

(deftype compile-option-t ()
  `(and keyword
	(member :default :fastest :compile-speed :debug :safety)))

(defun compile-option-form (option)
  (declare (type compile-option-t option))
  (case option
    (:default
     `(optimize (speed 3) (safety 1)))
    (:fastest
     `(optimize (speed 3) (safety 0)))
    (:compile-speed
     `(optimize (speed 0) (safety 0) (compilation-speed 3)))
    (:safety
     `(optimize (safety 3)))
    (:debug ;; Compiling would be suuuuuper slow
     `(optimize (safety 3) (debug 3)))))


(defun translate-adjustable-shape (shape) ;; tensor-input-shape
  "
## [function] translate-adjustable-shape

Reading the *adjustable-shape-table*, the function returns an list consisted of fixnum.

If there's any undetermined one, returns an error (TODO: Add Conditions)"
  (declare (type list shape)
	   (optimize (speed 3)))
  ;;(map 'list #'read-symbol shape)
  (map 'list #'cl-waffe2/vm:maybe-observe-axis shape))

(declaim (ftype (function ((or fixnum symbol)) fixnum) read-adjustable-symbol))
(defun read-adjustable-symbol (s)
  (typecase s
    (fixnum s)
    (symbol
     (or
      (let ((out (read-symbol s)))
	(when (integerp out) out))
      (when (cl-waffe2/vm:symbol-lazyaxis s)
	(cl-waffe2/vm:observe-axis
	 (cl-waffe2/vm:symbol-lazyaxis s)))
      (error "read-adjustable-symbol: encountered unknown symbol: ~a
~a"
	     s
	     (cl-waffe2/vm::render-debug-info))))))


(defmacro with-adjustable-symbol ((symbol-name symbol-value) &body body)
  "
Registering adjustable symbol (a.k.a: dynamic shape) to the runtime. This variable can be observed via cl-waffe2/vm:maybe-observe-axis.

Usage:

(with-adjustable-symbols ((a 1) (b 1))
    (with-let-adjustable-symbols (a b)
        (print a)   ;; = 1
        (print b))) ;; = 1
"

  `(let* ((*adjustable-shape-table* (or *adjustable-shape-table* (make-hash-table))))
     (setf (gethash ',symbol-name *adjustable-shape-table*) ,symbol-value)
     ;; [TODO/Fixme?]
     ;; When quitting the scope, this macro should delete the symbol-name from adjustable-symbol-table.
     ,@body))

(defmacro with-adjustable-symbols ((&rest forms) &body body)
  (labels ((expand-form (rest-forms)
	     (if (null rest-forms)
		 `(progn ,@body)
		 `(with-adjustable-symbol (,@(car rest-forms))
		    ,(expand-form (cdr rest-forms))))))
    (expand-form forms)))

(defmacro with-adjustable-symbol-scope (&body body)
  `(let* ((*adjustable-shape-table* (alexandria:copy-hash-table (or *adjustable-shape-table* (make-hash-table)))))
     ,@body))

(defun register-adjustable-shape (symbol value)
  (setf (gethash symbol *adjustable-shape-table*) value))

(defmacro with-let-adjustable-symbol (symbol-name &body body)
  ;; TO DELETE: Binding with symbol-name
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
  (declare (optimize (speed 3) (safety 0)))
  (if *adjustable-shape-table*
      (typecase symbol
	(symbol
	 ;; out <- table[symbol]
	 (let ((out (gethash symbol *adjustable-shape-table*)))
	   (if (null out)
	       symbol ;; No result?
	       (if (and (symbolp out)
			(not (eq symbol out))) ;; A -> A ...
		   ;; A = BATCH_SIZE = 10
		   (read-symbol out)
		   out))))
	;; Return fixnum
	(T symbol))
      symbol))

(defun range (from to)
  (loop for i fixnum upfrom from below to collect i))

(define-compiler-macro range (from to)
  `(loop for i fixnum upfrom ,from below ,to collect i))

(defun sync (list order) (loop for o in order collect (nth o list)))

(define-compiler-macro sync (list order) `(loop for o in ,order collect (nth o ,list)))

(defun find-size (wtensors rank)
  ;; Tries to find a number of shape from the same rank
  ;; If failed, returns a symbol.
  (nth rank
       (wtensor-shape
	(or
	 (find t wtensors
	       :test #'(lambda (_ wtensor)
			 (declare (ignore _))
			 (numberp (nth rank (wtensor-shape wtensor)))))
	 (car wtensors)))))

(defun l* (&rest args)
  (if (= (length args) 1)
      (car args)
      (apply #'* args)))

(defun subtype-equal (t1 t2)
  (or (subtypep t1 t2)
      (subtypep t2 t1)))

(defun symbol-eq (x y)
  (and
   (symbolp x)
   (symbolp y)
   (equal (symbol-name x)
	  (symbol-name y))))

(defmacro with-flexible-shape ((bind shape) &body body)
  "body -> returning tensor"
  (alexandria:with-gensyms (pos)
    `(let* ((,pos   (position '~ ,shape :test #'symbol-eq))
	    (,bind (loop for s in ,shape unless (symbol-eq s '~) collect s))
	    (created-tensor (progn ,@body)))
       (if ,pos
	   (cl-waffe2/base-impl:!flexible created-tensor :at ,pos)
	   created-tensor))))
       

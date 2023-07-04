
(in-package :cl-waffe2/vm.generic-tensor)

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
    (symbol `(the fixnum (read-symbol ',x)))
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


(defun make-clone (tensor &optional name)
  (let* ((out (make-input (shape tensor) (or name nil)
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
    out))

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


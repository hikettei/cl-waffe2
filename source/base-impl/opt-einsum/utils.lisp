
(in-package :cl-waffe2/base-impl)


(defun symbol-eq (x y)
  (and
   (symbolp x)
   (symbolp y)
   (equal (symbol-name x)
	  (symbol-name y))))


(defun make-symbol->idx-table (shapes)
  (let ((out (make-hash-table)))
    (loop with count fixnum = 0
	  for s in shapes
	  unless (gethash s out)
	    do (setf (gethash s out) count)
	       (incf count))
    out))

(defun symbol->idx (symbol table)
  (gethash symbol table))

(defun shape->ids (shape table &key
				 (broadcast '~)
				 (declared nil))
  (loop for s in shape
	collect
	(cond
	  ((find s declared :test #'symbol-eq)
	   nil)
	  ((symbol-eq s broadcast)
	   s)
	  (T
	   (symbol->idx s table)))))

(defun compose (&rest fns)
  (if fns
      (let ((fn1 (car (last fns)))
            (fns (butlast fns)))
        #'(lambda (&rest args)
                   (reduce #'funcall fns
                           :from-end t
                           :initial-value (apply fn1 args))))
      #'identity))

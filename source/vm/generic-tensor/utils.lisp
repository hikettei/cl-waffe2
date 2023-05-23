
(in-package :cl-waffe2/vm.generic-tensor)

(declaim (ftype (function (cons fixnum) cons) fill-with-d))
(defun fill-with-d (shape i)
  (declare (optimize (speed 3))
	   (type cons shape)
	   (type fixnum i))
  (let ((index -1))
    (declare (type fixnum index))
    (map 'list (lambda (x)
		 (declare (ignore x))
		 (incf index 1)
		 (cond
		   ((= i index)
		    1)
		   (T 0)))
	 shape)))

(declaim (ftype (function (list fixnum) fixnum) get-stride))
(defun get-stride (shape dim)
  (declare (optimize (speed 3) (safety 0))
	   (type list shape)
	   (type fixnum dim))
  (let ((subscripts (fill-with-d shape dim)))
    (apply #'+ (maplist #'(lambda (x y)
			    (the fixnum
				 (* (the fixnum (car x))
				    (the fixnum (apply #'* (cdr y))))))
			subscripts
			shape))))

(declaim (ftype (function (list) list)
		column-major-calc-strides
		row-major-calc-strides))
(defun column-major-calc-strides (shapes)
  (declare (optimize (speed 3))
	   (type list shapes))
  (loop for i fixnum upfrom 0 below (length shapes)
	collect (get-stride shapes i)))

(defun row-major-calc-strides (shape)
  (declare (optimize (speed 3))
	   (type list shape))
  (let* ((num-dims (length shape))
         (strides (make-list num-dims :initial-element 1)))
    (loop for i from 1 to (- num-dims 1) do
      (setf (nth i strides) (the fixnum
				 (* (the fixnum (nth (- i 1) strides))
				    (the fixnum (nth (- i 1) shape))))))
    strides))

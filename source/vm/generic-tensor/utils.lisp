
(in-package :cl-waffe2/vm.generic-tensor)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

(defun lazy* (x y)
  (if (and (typep x 'number)
	   (typep y 'number))
      (* x y)
      `(* ,x ,y)))

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


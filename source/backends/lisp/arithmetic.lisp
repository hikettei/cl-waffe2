
(in-package :cl-waffe2/backends.lisp)

;; define-with-typevar
(defun singele-float-add (x y offsetx offsety size)
  (declare (optimize (speed 3))
	   (type (simple-array single-float (*)) x y)
	   (type fixnum offsetx offsety size))
  (dotimes (i size)
    (incf (aref x (+ offsetx i)) (aref y (+ offsety i)))))


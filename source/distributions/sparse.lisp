
(in-package :cl-waffe2/distributions)

(defun ax+b (a x-start x-end b)
  (loop for x fixnum upfrom x-start below x-end
	collect (+ (* a x) b)))


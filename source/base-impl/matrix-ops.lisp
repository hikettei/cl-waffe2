
(in-package :cl-waffe2/base-impl)

(defnode (MatMulNode (myself &key transpose-a transpose-b)
	  :where `(A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :slots ((transpose-a :initarg :transpose-a :type boolean :reader trans-a?)
		  (transpose-b :initarg :transpose-b :type boolean :reader trans-b?))
	  :documentation ""))

(defun !matmul (x y &key (out nil) (transpose-x nil) (transpose-y nil))
  "X[~ i j] @ Y[~ j k] -> C[~ i k]"
  (let* ((i  (nth (if transpose-x 1 0) (last (shape x) 2)))
	 (jx (nth (if transpose-x 0 1) (last (shape x) 2)))
	 (jy (nth (if transpose-y 1 0) (last (shape y) 2)))
	 (k  (nth (if transpose-y 0 1) (last (shape y) 2)))
	 (out (or out (make-input `(,@(butlast (shape x) 2) ,i ,k) nil
				  :dtype (dtype x)
				  :order (order x)))))

    (assert (= jx jy) nil "Assertion Failed with X[~ i j] @ Y[~ j k] -> C[~ i k]. ~a and ~a" (shape x) (shape y))

    (forward (MatmulNode :transpose-a transpose-x
			 :transpose-b transpose-y)
	     x
	     y
	     out)))



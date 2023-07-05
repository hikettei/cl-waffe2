
(in-package :cl-waffe2/optimizers)

#|
(defmacro defoptimizer (name))

(defoptimier SGD ()
  :slots
  :step ((parameter)
	 |#

(defun composed-node (x y)
  (!mul
   (!sin (!add x y))
   (!cos (!add x y))))

(defmodel (ComposedFunction (self)
	   :where (A[~] B[~] -> [~])
	   :on-call-> ((self x y)
		       (declare (ignore self))
		       (composed-node x y))))

(define-composite-function (ComposedFunction) !composed)

(defmodel (Sin-Inlined (self)
	   :where (X[~] OUT[~] -> [~])
	   :on-call-> ((self x out)
		       (declare (ignore self))
		       (forward (SinNode) x out))))

(define-composite-function (Sin-Inlined) !sin-inline)



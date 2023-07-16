
(in-package :cl-waffe2/vm.nodes.test)

(defun M= (tensor1 tensor2)
  (every #'= (tensor-vec tensor1) (tensor-vec tensor2)))

(defmodel (SumUpComposite (self)
	   :where ([~] -> [scal] where scal = 1)
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (!sum x))))

(defmodel (SumUp2DComposite (self)
	   :where (X[a b] -> [scal] where scal = 1)
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (!sum x))))

;; This case: The dispatching of function settles on a pattern
(define-composite-function (SumUp2DComposite) !sumup-2d-float :dtype :float)

(define-composite-function (SumUpComposite) !sumup-float :dtype :float)

(define-composite-function (SumUpComposite) !sumup-generic)


(defun composite-fundamental-function-test ()
  (let ((a (randn `(10 10))))
    (M= (proceed (!sum a))
	(!sumup-2d-float a))))

(defun composite-rank-dispatch (rank)
  (let* ((shape (loop for r upfrom 0 below rank
		      collect 10))
	 (a (randn shape)))
    (M= (proceed (!sum a))
	(!sumup-float a))))

(defun composite-dtype-dispatch (dtype)
  (let* ((a (ax+b `(10 10) 0 1 :dtype dtype)))
    (M= (proceed (!sum a))
	(!sumup-generic a))))

(test composite-function
  (is (composite-fundamental-function-test)))

(test composite-rank-dispatching-test
  (is (composite-rank-dispatch 1))
  (is (composite-rank-dispatch 2))
  (is (composite-rank-dispatch 3))
  (is (composite-rank-dispatch 4))
  (is (composite-rank-dispatch 5))
  (is (composite-rank-dispatch 6))
  (is (composite-rank-dispatch 7))
  (is (composite-rank-dispatch 8))
  (is (composite-rank-dispatch 9)))

(test composite-dtype-dispatching-test
  (is (composite-dtype-dispatch :double))
  (is (composite-dtype-dispatch :float))
  (is (composite-dtype-dispatch :uint32))
  (is (composite-dtype-dispatch :int32))
  (is (composite-dtype-dispatch :int16))
  (is (composite-dtype-dispatch :uint16))
  (is (composite-dtype-dispatch :uint8))
  (is (composite-dtype-dispatch :int8)))
  

;; Complex Node Test

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

;; Add: Dtype Checker
;; どこかのCopyがResetされていない・・・？
(defun test-composed-function (a b a1 b1)
    (M= (proceed (composed-node a b))
        (!composed a1 b1)))

;; No Side Effects?
(test test-composed-function
  (is (test-composed-function
       (ax+b `(3 3) 0 1)
       (ax+b `(3 3) 0 1)
       (ax+b `(3 3) 0 1)
       (ax+b `(3 3) 0 1)
       ))
  (is (test-composed-function
       (ax+b `(3 3) 0 1)
       (ax+b `(3 3) 0 1)
       (ax+b `(3 3) 0 1)
       (ax+b `(3 3) 0 1)
       ))
  (is (test-composed-function
       (ax+b `(3 3) 0 1)
       (ax+b `(3 3) 0 1)
       (ax+b `(3 3) 0 1)
       (ax+b `(3 3) 0 1)
       )))

;; composite-function -> composite-function Test


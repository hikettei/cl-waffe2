
(in-package :cl-waffe2/vm.nodes.test)

(defun M= (tensor1 tensor2)
  (every #'= (tensor-vec tensor1) (tensor-vec tensor2)))

(defmodel (SinModel (self) :on-call-> ((self x) (declare (ignore self)) (cl-waffe2/base-impl:!sin x))))
(defmodel-as (SinModel) :where (X[~] -> Out[~]) :asif :function :named sin-static1)

(defmodel (Softmax2DTestModel (self)
	   :where (A[x y] -> OUT[x y])
	   :on-call-> ((self a)
		       (declare (ignore self))
		       (!softmax a))))

(defmodel-as (Softmax2DTestModel)
  :asif :function
  :named softmax-2d-f)

(defmodel-as (Softmax2DTestModel)
  :asif :node
  :named !softmax-2d-f)

(defmodel (SumUpModel (self)
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (->scal (!sum x)))))

(defmodel (MeanUpModel (self)
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (->scal (!mean x)))))

(defmodel-as (SumUpModel)  :where (A[~] -> out[one] where one = 1) :named sumup-static :asif :function)
(defmodel-as (MeanUpModel) :where (A[~] -> out[one] where one = 1) :named meanup-static :asif :function)

(defmodel-as (SumUpModel)  :where (A[~] -> out[one] where one = 1) :named !sumup-static :asif :node :differentiable t)
(defmodel-as (MeanUpModel) :where (A[~] -> out[one] where one = 1) :named !meanup-static :asif :node :differentiable t)

(defmodel-as (SinModel) :where (A[~] -> B[~]) :named !sinmodel :asif :node :differentiable t)

(deftest defmodel->function-reduction-test
  (ok (= 100 (tensor-vec (sumup-static  (ax+b `(10 10) 0 1)))))
  (ok (= 1   (tensor-vec   (meanup-static (ax+b `(10 10) 0 1))))))

(deftest defmodel->function-test
  (ok (M= (proceed (!sin (ax+b `(10 10) 0 1)))
	  (sin-static1 (ax+b `(10 10) 0 1))))
  (ok (M= (proceed (!softmax (ax+b `(10 10) 0 1)))
	  (softmax-2d-f (ax+b `(10 10) 0 1))))
  (ok (M= (proceed (!sin (ax+b `(10 10 10) 0 1)))
	  (sin-static1 (ax+b `(10 10 10) 0 1))))
  (ok (M= (proceed (!softmax (ax+b `(10 5) 0 1)))
	  (softmax-2d-f (ax+b `(10 5) 0 1)))))

(deftest defmodel->node-forward
  (ok (= 100 (tensor-vec (proceed (!sumup-static (ax+b `(10 10) 0 1))))))
  (ok (= 1 (tensor-vec (proceed (!meanup-static (ax+b `(10 10) 0 1))))))
  (ok (M= (proceed (!softmax      (ax+b `(10 10) 0 1)))
	  (proceed (!softmax-2d-f (ax+b `(10 10) 0 1))))))

(defun defmodel->node-diff-1 ()
  (let ((a (parameter (ax+b `(3 5) 0 1))))
    (proceed-backward
     (!sinmodel a))
    (< (- (cos 1) (mref (grad a) 0 0)) 0.00001)))

(defun defmodel->node-diff-1-vm ()
  (let ((a (parameter (ax+b `(3 5) 0 1))))
    (let ((model (build (!sinmodel a))))
      (forward model)
      (backward model))
    (< (- (cos 1) (mref (grad a) 0 0)) 0.00001)))

(defun defmodel->node-diff-2 ()
  (let ((a (parameter (ax+b `(3 5) 0 1))))
    (proceed-backward
     (!sumup-static a))
    (= 1 (vref (grad a) 0))))

;; Backward tests
(deftest defmodel-as-diff-test
  (ok (defmodel->node-diff-1))
  (ok (defmodel->node-diff-1-vm))
  (ok (defmodel->node-diff-2)))

(node->defnode !softmax-jit-lazy (A[~] -> B[~])
  (!softmax a))

(deftest node->defnode-test
  (ok
   (M=
    (proceed (!softmax (ax+b `(20 20) 0 2)))
    (proceed (!softmax-jit-lazy (ax+b `(20 20) 0 2))))))

(node->defnode !gelu-jit-lazy (A[~] -> B[~])
  (!gelu a))

(deftest defmodel-as-node-diff-test-GeLU
  (ok
   (let ((a (parameter (ax+b `(10 10) 0.001 2)))
	 (b (parameter (ax+b `(10 10) 0.001 2))))
     (proceed-backward
      (!mean (!gelu-jit-lazy a)))
     (proceed-backward
      (!mean (!gelu-jit-lazy b)))
     (and
      (not (= (vref (grad a) 0) 0.0))
      (M= (grad a) (grad b))))))

(defun meet-with-dynamic-shape-larger ()
  (let ((model (build
		(!gelu-jit-lazy (make-input `(A B) :X))
		:inputs `(:X))))
    ;; Every time Reallocation
    (forward model (ax+b `(1 10) 0 2))
    (forward model (ax+b `(2 10) 0 1))
    (forward model (ax+b `(3 10) 0 2))
    (every #'(lambda (x) (= x 0.841192))
	   (tensor-vec (forward model (ax+b `(4 10) 0 1))))))

(defun meet-with-dynamic-shape-smaller ()
  (let ((model (build
		(!gelu-jit-lazy (make-input `(A B) :X))
		:inputs `(:X))))
    ;; Re-using allocations
    (forward model (ax+b `(4 10) 0 2))
    (forward model (ax+b `(3 10) 0 1))
    (forward model (ax+b `(2 10) 0 2))
    (let ((out (forward model (ax+b `(1 10) 0 1))))
      (every #'(lambda (nth)
		 (= (vref out nth) 0.841192))
	     (loop for i upfrom 0 below 10 collect i)))))

(defun meet-with-dynamic-shape-complicated ()
  (let ((model (build
		(!relu (!gelu-jit-lazy (!relu (make-input `(A B) :X))))
		:inputs `(:X))))
    ;; Re-using allocations
    (forward model (ax+b `(4 10) 0 2))
    (backward model)
    (forward model (ax+b `(3 10) 0 1))
    (forward model (ax+b `(2 10) 0 2))
    (let ((out (forward model (ax+b `(1 10) 0 1))))
      (every #'(lambda (nth)
		 (= (vref out nth) 0.841192))
	     (loop for i upfrom 0 below 10 collect i)))))

(deftest meet-with-dynamic-shape
  (ok (meet-with-dynamic-shape-larger))
  (ok (meet-with-dynamic-shape-smaller))
  (ok (meet-with-dynamic-shape-complicated)))

;; !sin a !sin b was computed at once.
(node->defnode !arith-test (A[~] B[~] -> C[~])
  (!add (!sin a) (!sin b)))

;; Confirmed !arith-test backward was called at once.
(defun backward-route-test ()
  (let ((a (parameter (ax+b `(10 10) 0 2)))
	(b (parameter (ax+b `(10 10) 0 2))))
    (proceed-backward (!arith-test a b))
    ;;(cl-waffe2/vm:disassemble-waffe2-ir (!arith-test a b))
    (list (grad a) (grad b))))

(deftest defmodel-as-backward-route-test
  (ok (backward-route-test)))



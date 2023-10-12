
(in-package :cl-waffe2/vm.nodes.test)

(in-suite :test-nodes)

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

(test defmodel->function-reduction-test
  (is (= 100 (tensor-vec (sumup-static  (ax+b `(10 10) 0 1)))))
  (is (= 1   (tensor-vec   (meanup-static (ax+b `(10 10) 0 1))))))

(test defmodel->function-test
  (is (M= (proceed (!sin (ax+b `(10 10) 0 1)))
	  (sin-static1 (ax+b `(10 10) 0 1))))
  (is (M= (proceed (!softmax (ax+b `(10 10) 0 1)))
	  (softmax-2d-f (ax+b `(10 10) 0 1))))
  (is (M= (proceed (!sin (ax+b `(10 10 10) 0 1)))
	  (sin-static1 (ax+b `(10 10 10) 0 1))))
  (is (M= (proceed (!softmax (ax+b `(10 5) 0 1)))
	  (softmax-2d-f (ax+b `(10 5) 0 1)))))

(test defmodel->node-forward
  (is (= 100 (tensor-vec (proceed (!sumup-static (ax+b `(10 10) 0 1))))))
  (is (= 1 (tensor-vec (proceed (!meanup-static (ax+b `(10 10) 0 1))))))
  (is (M= (proceed (!softmax      (ax+b `(10 10) 0 1)))
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
(test defmodel-as-diff-test
  (is (defmodel->node-diff-1))
  (is (defmodel->node-diff-1-vm))
  (is (defmodel->node-diff-2)))

;; 1. もうちょっとdefmodel->node-diffのテスト増やすべき
;; 2. Backward opt
;; 3. connected with dynamic-shape?
;; 4. node->defnode

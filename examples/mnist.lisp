(in-package :cl-user)

(defpackage :mnist-example
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/nn
   :cl-waffe2/distributions
   :cl-waffe2/base-impl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes))

(in-package :mnist-example)

;; Things to add:
;; !matmul !t !t backward tests
;; defoptimizer/deftrainer
;; Numpy tensor loadder
;; Training MLP

;; To ADD:
;; call->
;; sequence
;; deftrainer

;; TO ADD:
;; (call-> x layer1 layer2 layer3)
;; (x = (sequence (LinearLayer1 (LinearLayer2 ...

;; ugly...
;; call -> Node

;; one more ugly part: defmodel

#|
(defmodel (MLP-Model (self)
	   :slots ((layer1 :initarg :layer1 :accessor mlp-layer1)
		   (layer2 :initarg :layer2 :accessor mlp-layer2)
		   (layer3 :initarg :layer3 :accessor mlp-layer3))
	   :initargs (:layer1 (LinearLayer 784 512)
		      :layer2 (LinearLayer 512 256)
		      :layer3 (LinearLayer 256 10))
	   :on-call-> ((self x)
		       (call-> x
			       (slot-value self 'layer1)
			       (asnode #'!tanh)
			       (slot-value self 'layer2)
			       (asnode #'!tanh)
			       (slot-value self 'layer3)
			       (asnode #'!softmax)))))
|#

(defsequence MLP-Sequence (in-features hidden-dim out-features
			   &key (activation #'!relu))
	     "Three Layers MLP Model"
	     (LinearLayer in-features hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim out-features))

;; TODO
;; 1. TO FIX: forward with batch-size (save-for-backward) (OK)
;; 2. TODO: criterion.lisp
;; 3. TODO: cl-waffe2/nn
;; 4. TODO: defoptimizer
;; 5. TODO: Documentations, Slides.


(deftrainer (MLPTrainer (self in-class out-class
			      &key
			      (hidden-size 256)
			      (activation #'!tanh))
	     :model     (MLP-Sequence in-class hidden-size out-class :activation activation)
	     :optimizer (cl-waffe2/optimizers:SGD :lr 1e-2)
	     :build ((self)
		     (let ((out (!sum (softmax-cross-entropy
				       (call
					(model self)
					(make-input `(batch-size ,in-class)  :X))
				       (make-input `(batch-size  ,out-class) :Y)))))

		       
		       out))
	     :minimize! ((self)
			 (zero-grads! (model self))
			 (let ((loss (forward     (model self))))
			   (format t "Loss: ~a~%" (tensor-vec loss)))
			 (backward    (model self))
			 (optimize!   (model self)))
	     :set-inputs ((self x y)
			  (set-input (model self) :X x)
			  (set-input (model self) :Y y))
	     :predict ((self x)
		       (call (model self) x))))

(deftrainer (MLPTrainer-Test (self in-class out-class
			      &key
			      (hidden-size 256)
			      (activation #'!tanh))
	     :model     (MLP-Sequence in-class hidden-size out-class :activation activation)
	     :optimizer (cl-waffe2/optimizers:SGD :lr 1e-2)
	     :build ((self)
		     (let ((out (!sum (!mul
				 (call
				  (model self)
				  (make-input `(batch-size ,in-class)  :X))
				 (make-input `(batch-size  ,out-class) :Y)))))

		       
		       out))
	     :minimize! ((self)
			 (zero-grads! (model self))
			 (let ((loss (forward     (model self))))
			   (format t "Loss: ~a~%" (tensor-vec loss)))
			 (backward    (model self))
			 (optimize!   (model self)))
	     :set-inputs ((self x y)
			  (set-input (model self) :X x)
			  (set-input (model self) :Y y))
	     :predict ((self x)
		       (call (model self) x))))

;; TODO: Batch-Size 10 -> 1

(defun perform-test ()
  (let ((trainer (MLPTrainer 50 10 :hidden-size 30 :activation #'!tanh)))
    (set-inputs trainer (randn `(3 50)) (bernoulli `(3 10) 0.1))
    
    (minimize!  trainer)
    ;;(minimize!  trainer)
    ;;(minimize!  trainer)
    

   ;; (time
   ;;  (progn
       ;;(set-inputs trainer (randn `(10 784)) (randn `(10 10)))
   ;;    (minimize!  trainer)
   ;;    ))    
    trainer))


;; Add this code to test cases
(defun test-softmax ()
  (let ((a (parameter (bernoulli `(10 10) 0.3)))
	(b (bernoulli `(10 10) 0.3)))
    (proceed-backward (!sum (softmax-cross-entropy (!relu a) b)))
    (print a)
    (grad a)))


(in-package :cl-user)

(load "./cl-waffe2.asd")
(ql:quickload :cl-waffe2)

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

(defsequence MLP-Sequence (in-features hidden-dim out-features
			   &key (activation #'!relu))
	     "Three Layers MLP Model"
	     (LinearLayer in-features hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim out-features))


(deftrainer (MLPTrainer (self in-class out-class
			      &key
			      (hidden-size 256)
			      (activation #'!tanh))
	     :model     (MLP-Sequence in-class hidden-size out-class :activation activation)
	     :compile-mode :fastest
	     :optimizer (cl-waffe2/optimizers:Adam :lr 1e-3)
	     :build ((self)
		     (let ((out (!mean (softmax-cross-entropy
					(call
					 (model self)
					 (make-input `(batch-size ,in-class)  :X))
					(make-input  `(batch-size ,out-class) :Y)))))
		       
		       out))
	     :minimize! ((self)
			 (zero-grads! (model self))
			 (let ((loss (forward     (model self))))
			   (format t "Training loss: ~a~%" (aref (tensor-vec loss) 0)))
			 (backward    (model self))
			 (optimize!   (model self)))
	     :set-inputs ((self x y)
			  (set-input (model self) :X x)
			  (set-input (model self) :Y y))
	     :predict ((self x)
		       (call (model self) x))))

;; TODO
;; Optimizers: Momentum Adagrad RMSProp Adam
;; Documentations on defsequence, deftrainer
;; 


(defun train-and-valid-mnist ()
  (let ((trainer (MLPTrainer 784 10 :hidden-size 256 :activation #'!relu)))
    
    (time
     (dotimes (i 1000)
       (set-inputs trainer (randn `(100 784)) (bernoulli `(100 10) 0.3))
       (minimize!  trainer)))
    trainer))

(train-and-valid-mnist)


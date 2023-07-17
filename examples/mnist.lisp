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
	     :compile-mode :default
	     :optimizer (cl-waffe2/optimizers:SGD :lr 1e-2)
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
			   (format t "Loss: ~a~%" (tensor-vec loss)))
			 (backward    (model self))
			 (optimize!   (model self))
			 )
	     :set-inputs ((self x y)
			  (set-input (model self) :X x)
			  (set-input (model self) :Y y))
	     :predict ((self x)
		       (call (model self) x))))
;; TODO: Batch-Size 10 -> 1

(defun perform-test ()
  (let ((trainer (MLPTrainer 50 10 :hidden-size 30 :activation #'!relu)))
    
    (set-inputs trainer (randn `(3 50)) (bernoulli `(3 10) 0.1))
    
    (minimize!  trainer)

    (set-inputs trainer (randn `(3 50)) (bernoulli `(3 10) 0.1))
    
    (minimize!  trainer)

    ;;(set-inputs trainer (randn `(3 50)) (bernoulli `(3 10) 0.1))
    
    ;;(minimize!  trainer)
    

   ;; (time
   ;;  (progn
       ;;(set-inputs trainer (randn `(10 784)) (randn `(10 10)))
   ;;    (minimize!  trainer)
   ;;    ))    
    trainer))

(perform-test)


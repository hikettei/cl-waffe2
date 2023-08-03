
(in-package :cl-user)

(defpackage :cl-waffe2-example1
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/nn
   :cl-waffe2/distributions
   :cl-waffe2/base-impl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/optimizers

   :cl-waffe2/backends.jit.cpu
   :cl-waffe2/backends.cpu
   ))

(in-package :cl-waffe2-example1)

(defsequence Simple-MLP (in-features hidden-dim)
	     (LinearLayer in-features hidden-dim t)
	     (asnode #'!sigmoid)
	     (LinearLayer hidden-dim 1 t))

;; Naming:... set-input VS set-inputs
;; allow: (forward self)
;; make trainer forwardable

(deftrainer (MLPTrainer (self in-features hidden-dim &key (lr 1e-1))
	     :model (Simple-MLP in-features hidden-dim)
	     :optimizer (SGD :lr lr)
	     :compile-mode :fastest
	     :build ((self)
		     (MSE
		      (make-input `(batch-size 1) :TrainY)
		      (call (model self) (make-input `(batch-size ,in-features) :TrainX))))
	     
	     :set-inputs ((self x y)
			  (set-input (model self) :TrainX x)
			  (set-input (model self) :TrainY y))
	     :minimize! ((self)
			 (zero-grads! (model self))
			 (let ((loss (forward (model self))))
			   (format t "Training Loss: ~a~%" (tensor-vec loss)))
			 (backward  (model self))
			 (optimize! (model self)))
	     :predict ((self x)
		       (call (model self) x))))


;; Training sin wave from random noises
;; If we forget to call proceed when making training data?
;; -> set-input must return error.
(defun train (&key
		(batch-size 100)
		(iter-num 3000))
  (let* ((X (proceed (!sin (ax+b `(,batch-size 100) 0.01 0.1))))
 	 (Y (proceed (!cos (ax+b `(,batch-size 1)   0.01 0.1))))
	 (trainer (MLPTrainer 100 10 :lr 1e-3)))
    
    (set-inputs trainer X Y)
    (time
     (loop for nth-epoch fixnum upfrom 0 below iter-num
	   do (minimize! trainer)))))

(with-devices (JITCPUTensor JITCPUScalarTensor CPUTensor)
  (train)) 




(in-package :cl-user)

;;
;; This example provides the smallest package for training neural network in cl-waffe2
;;

(defpackage :mlp-sin-wave
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/nn
   :cl-waffe2/distributions
   :cl-waffe2/base-impl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/optimizers

   :cl-waffe2/backends.cpu))

(in-package :mlp-sin-wave)

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Network Template: Criterion
(defun criterion (criterion X Y &key (reductions nil))
  (apply #'call->
	 (funcall criterion X Y)
	 (map 'list #'asnode reductions)))
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; Defines a model
(defsequence Simple-MLP (in-features hidden-dim)
	     (LinearLayer in-features hidden-dim t)
	     (asnode #'!sigmoid)
	     (LinearLayer hidden-dim 1 t))

;; Constructs/Compiles the neural network
(defun build-mlp-model (in-features hidden-dim &key (lr 1e-1))
  (let* ((lazy-loss (criterion #'MSE
			       (call (Simple-MLP in-features hidden-dim)
				     (make-input `(batch-size ,in-features) :TrainX))
			       (make-input `(batch-size 1) :TrainY)
			       :reductions (list #'!mean #'->scal)))
	 (model (build lazy-loss :inputs `(:TrainX :TrainY))))

    ;; Initializes and hooks AbstractOptimizers
    (mapc (hooker x (SGD x :lr lr)) (model-parameters model))
    model))

;; Calls forward/backward propagations, and optimizes.
(defun step-train (model train-x train-y)
  (let ((act-loss (forward model train-x train-y)))
    (format t "Loss = ~a~%" (tensor-vec act-loss)))
  (backward model)
  (mapc #'call-optimizer! (model-parameters model)))

;; Training sin wave from random noises
;; If we forget to call proceed when making training data?
;; -> set-input must return error.
(defun train (&key
		(batch-size 100)
		(iter-num 3000))
  (let* ((X     (proceed (!sin (ax+b `(,batch-size 100) 0.01 0.1))))
 	 (Y     (proceed (!cos (ax+b `(,batch-size 1)   0.01 0.1))))
	 (model (build-mlp-model 100 10 :lr 1e-3)))
    
    (time
     (loop for nth-epoch fixnum upfrom 0 below iter-num
	   do (step-train model X Y)))

    ;; Displays the compiled model
    (print model)))

;; Start Training:
(train)



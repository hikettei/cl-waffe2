
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
			      (activation #'!relu)
			      (lr 1e-3))
	     :model     (MLP-Sequence in-class hidden-size out-class :activation activation)
	     :compile-mode :fastest
	     :optimizer (cl-waffe2/optimizers:SGD :lr lr)
	     :build ((self)
		     (!mean (softmax-cross-entropy
			     (call
			      (model self)
			      (make-input `(batch-size ,in-class)  :X))
			     (make-input  `(batch-size ,out-class) :Y))))
	     :minimize! ((self)
			 (zero-grads! (compiled-model self))
			 (let ((loss (forward     (compiled-model self))))
			   (format t "Training loss: ~a~%" (aref (tensor-vec loss) 0)))
			 (backward    (compiled-model self))
			 (optimize!   (compiled-model self)))
	     :set-inputs ((self x y)
			  (set-input (compiled-model self) :X x)
			  (set-input (compiled-model self) :Y y))
	     :predict ((self x)
		       (!argmax (call (model self) x)))))


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

(defun train-and-valid-mlp (&key
			      (epoch-num 10))
  (let* ((model (MLPTrainer 784 10 :lr 1e-2))
	 ;; Flatten Inputs
	 (train-img  (proceed (!div (!reshape *train-data*  t (* 28 28)) 255.0)))
	 (test-img   (proceed (!div (!reshape *test-data*   t (* 28 28)) 255.0)))
	 (train-label *train-label*)
	 (test-label  *test-label*))
    (format t "[Log] Start Training...~%")
    (dotimes (nth-epoch epoch-num)
      (format t "~ath Epoch...~%" nth-epoch)
      (loop for batch fixnum upfrom 0 below 60000 by 100 do

	;; Set training data.
	(let ((end (+ batch 100)))
	  ;; :X = Train[batch:batch+100, :]
	  (set-inputs model
		      (view train-img   `(,batch ,end) t)
		      (view train-label `(,batch ,end) t)))

	(minimize! model)))

    ;; TODO: Validate, Trying Adam
    model))



(in-package :mnist-sample)

;; TODO: softmax-cross-entropy ... optimized to-onehot, without making a copy.

(defsequence MNIST-CNN (&key
			(out-channels1 4)
			(out-channels2 16))
	     (Conv2D 1 out-channels1 `(3 3))
	     (asnode #'!relu)     
	     (MaxPool2D    `(2 2))
	     (Conv2D out-channels1 out-channels2 `(5 5))
	     (asnode #'!relu)
	     (MaxPool2D `(2 2))
	     (asnode #'!reshape t (* 16 4 4)) 
	     (LinearLayer (* 16 4 4) 10))

(deftrainer (CNNTrainer (self N &key (lr 1e-3))
	     :model     (MNIST-CNN)
	     :compile-mode :fastest
	     :optimizer (cl-waffe2/optimizers:Adam :lr lr)
	     :set-inputs ((self x y)
			  (set-input (compiled-model self) :X x)
			  (set-input (compiled-model self) :Y y))
	     :build ((self)
		     ;; X: (N 1 H W)
		     ;; Y: (N 10) where n = batch-size, H, W=28, 28 respectively as for MNIST.
		     (!sum
		      (softmax-cross-entropy
		       (call
			(model self)
			(make-input `(,N 1 28 28) :X))
		       (make-input  `(,N      10) :Y))))
	     :minimize! ((self)
			 (zero-grads! (compiled-model self))
			 (let* ((out  (forward (compiled-model self)))
				(loss (vref out 0)))
			   (backward    (compiled-model self))			   
			   (optimize! (compiled-model self))
			   (/ loss 100)))
	     :predict ((self x) (!argmax (call (model self) x)))))

(defmethod accuracy ((self CNNTrainer) x y)
  (let* ((out   (!argmax (call (model self) x)))
	 (label (!argmax y))
	 (total (proceed (->scal (!sum (A=B out label))))))
    (float (/ (tensor-vec total) (nth 0 (shape out))))))

(defun train-and-valid-cnn (&key
			      (epoch-num 10))
  (let* ((model (CNNTrainer 100 :lr 1e-3)) 
	 ;; Flatten Inputs
	 (train-img  (proceed (!div (!reshape *train-data*  60000 1 28 28) 255.0))) ;; (60000 1 28 28)
	 (test-img   (proceed (!div (!reshape *test-data*   10000 1 28 28) 255.0))) ;; (10000 1 28 28)
	 (train-label *train-label*) ;; (60000 10)
	 (test-label  *test-label*)  ;; (10000 10)
	 (total-loss 0.0))
    (format t "[Log] Start Training...~%")
    (dotimes (nth-epoch epoch-num)
      (format t "~ath Epoch...~%" nth-epoch)
      ;;(time
      (loop for batch fixnum upfrom 0 below 60000 by 100 do

	;; Set training data.
	(let ((end (+ batch 100)))
	  ;; :X = Train[batch:batch+100, :]
	  (set-inputs model
		      ;; Instead, can't we increase offset?
		      (proceed (->contiguous (view train-img   `(,batch ,end))))
		      (proceed (->contiguous (view train-label `(,batch ,end))))))
	(incf total-loss (minimize! model)))
      (format t "Training Loss: ~a~%" (/ total-loss 600))
      (setq total-loss 0.0))

    (format t "Validating...~%")
    (with-no-grad
      (format t "Valid Accuracy: ~a~%" (accuracy model test-img test-label)))
    model))

;;(train-and-valid-cnn :epoch-num 10)


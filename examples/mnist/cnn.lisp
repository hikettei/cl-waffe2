
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

(defun build-cnn (N &key (lr 1e-3))
  (with-cpu-jit (CPUTensor LispTensor)
    (let* ((model (MNIST-CNN))
	   (lazy-loss (criterion #'softmax-cross-entropy
				 (call model (make-input `(,N 1 28 28) :X))
				 (make-input `(,N 10) :Y)
				 :reductions (list #'!sum #'->scal)))
	   (compiled-model (build lazy-loss :inputs `(:X :Y))))
      (mapc (hooker x (Adam x :lr lr)) (model-parameters compiled-model))
      (values compiled-model model))))

;;(with-cpu-jit (CPUTensor LispTensor)
;;  (format t "[INFO] Doing CNN benchmark...~%")
;;  (proceed-bench (call (MNIST-CNN) (randn `(100 1 28 28))) :backward t))

(defun step-cnn (model X Y)
  (let ((act-loss (forward model X Y)))
    (backward model)
    (mapc #'call-optimizer! (model-parameters model))
    (/ (tensor-vec act-loss) 100)))

(defmethod accuracy ((model MNIST-CNN) x y)
  (let* ((out   (!argmax (call model x)))
	 (label (!argmax y))
	 (total (proceed (->scal (!sum (A=B out label))))))
    (float (/ (tensor-vec total) (nth 0 (shape out))))))

(defun train-and-valid-cnn (&key
			      (epoch-num 10))
  (multiple-value-bind (compiled-model model) (build-cnn 100 :lr 1e-3)
    (let* ((train-img  (proceed (!div (!reshape *train-data*  60000 1 28 28) 255.0))) ;; (60000 1 28 28)
	   (test-img   (proceed (!div (!reshape *test-data*   10000 1 28 28) 255.0))) ;; (10000 1 28 28)
	   (train-label *train-label*) ;; (60000 10)
	   (test-label  *test-label*)  ;; (10000 10)
	   (total-loss 0.0))
      (format t "[Log] Start Training...~%")
      (dotimes (nth-epoch epoch-num)
	(format t "~ath Epoch...~%" nth-epoch)

	(time
	 (loop for batch fixnum upfrom 0 below 60000 by 100 do
	   ;; Set training data.
	   
	   ;; :X = Train[batch:batch+100, :]
	   (incf total-loss
		  (step-cnn compiled-model
			    ;; Instead, can't we increase offset?
			    (tensor-displace-to train-img batch)
			    (tensor-displace-to train-label batch)))))
	(format t "Training Loss: ~a~%" (/ total-loss 600))
	(setq total-loss 0.0))
      (format t "Validating...~%")
      (with-no-grad
	(format t "Valid Accuracy: ~a~%" (accuracy model test-img test-label)))
      compiled-model)))

;;(train-and-valid-cnn :epoch-num 10)


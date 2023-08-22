
(in-package :mnist-sample)

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
	     :optimizer (cl-waffe2/optimizers:Adam :lr lr)
	     :build ((self)
		     (!sum (softmax-cross-entropy
			    (call
			     (model self)
			     (make-input `(batch-size ,in-class)  :X))
			    (make-input  `(batch-size ,out-class) :Y))))
	     :minimize! ((self)
			 (zero-grads! (compiled-model self))
			 (let* ((out  (forward (compiled-model self)))
				(loss (vref out 0)))
			   (backward    (compiled-model self))
			   (progn;;with-cpu-jit ()
			     (optimize! (compiled-model self))
			     (/ loss 100))))
	     :set-inputs ((self x y)
			  (set-input (compiled-model self) :X x)
			  (set-input (compiled-model self) :Y y))
	     :predict ((self x)
		       (!argmax (call (model self) x)))))

(defmethod accuracy ((self MLPTrainer) x y)
  (let* ((out   (!argmax (call (model self) x)))
	 (label (!argmax y))
	 (total (proceed (->scal (!sum (A=B out label))))))
    (float (/ (tensor-vec total) (nth 0 (shape out))))))

;; TODO: Adam, Dropout
;; Goal: MNIST 96~7% with MLP. (OK)

#|
For benchmarking the training time without loading training data
(defun train-and-valid-mlp (&key
			      (epoch-num 10))
  (let* ((model (MLPTrainer 784 10 :lr 1e-3)) ;; lr = 1e-2 for SGD
	 ;; Flatten Inputs
	 (train-img  (proceed (!div (!reshape *train-data*  t (* 28 28)) 255.0)))
	 (test-img   (proceed (!div (!reshape *test-data*   t (* 28 28)) 255.0)))
	 (train-label *train-label*)
	 (test-label  *test-label*)
	 (total-loss 0.0))
    (format t "[Log] Start Training...~%")
    (dotimes (nth-epoch epoch-num)
      (format t "~ath Epoch...~%" nth-epoch)

      (let ((batch 0))
      (let ((end (+ batch 100)))
	;; :X = Train[batch:batch+100, :]
	(set-inputs model
		    ;; Instead, increase offset?
		    (proceed (->contiguous (view train-img   `(,batch ,end) t)))
		    (proceed (->contiguous (view train-label `(,batch ,end) t)))

		    )))
      ;;(time
      (time
      (loop for batch fixnum upfrom 0 below 60000 by 100 do

	;; Set training data.
	;;(let ((end (+ batch 100)))
	  ;; :X = Train[batch:batch+100, :]
	  ;;(set-inputs model
		      ;; Instead, increase offset?
	;;	      (proceed (->contiguous (view train-img   `(,batch ,end) t)))
	;;	      (proceed (->contiguous (view train-label `(,batch ,end) t)))

	;;	      ))
	(incf total-loss (minimize! model))))
      (format t "Training Loss: ~a~%" (/ total-loss 600))
      (setq total-loss 0.0))

    ;; TODO: Validate, Trying Adam
    (with-no-grad
      (format t "Valid Accuracy: ~a~%" (accuracy model test-img test-label)))
model))
|#

(defun train-and-valid-mlp (&key
			      (epoch-num 10))
  (let* ((model (MLPTrainer 784 10 :lr 1e-3)) ;; lr = 1e-2 for SGD
	 ;; Flatten Inputs
	 (train-img  (proceed (!div (!reshape *train-data*  t (* 28 28)) 255.0)))
	 (test-img   (proceed (!div (!reshape *test-data*   t (* 28 28)) 255.0)))
	 (train-label *train-label*)
	 (test-label  *test-label*)
	 (total-loss 0.0))
    (format t "[Log] Start Training...~%")
    (dotimes (nth-epoch epoch-num)
      (format t "~ath Epoch...~%" nth-epoch)

      
      (loop for batch fixnum upfrom 0 below 60000 by 100 do	 
	(let ((end (+ batch 100)))
	  ;; :X = Train[batch:batch+100, :]
	  (set-inputs model
		      ;; Instead, increase offset?
		      (proceed (->contiguous (view train-img   `(,batch ,end) t)))
		      (proceed (->contiguous (view train-label `(,batch ,end) t)))))
	(incf total-loss (minimize! model)))
      (format t "Training Loss: ~a~%" (/ total-loss 600))
      (setq total-loss 0.0))

    ;; TODO: Validate, Trying Adam
    (with-no-grad
      (format t "Valid Accuracy: ~a~%" (accuracy model test-img test-label)))
    model))

;; PyTorch ... 7sec
;; 1Epoch 13s(cl-waffe2) -> 6s (cl-waffe)
;;(progn;;with-cpu-jit (CPUTensor LispTensor)
;;  (format t "MLP(hidden_size=256) optimizer=Adam batch-size=100~%")
;;  (train-and-valid-mlp :epoch-num 10))




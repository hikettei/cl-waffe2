(in-package :mnist-sample)


;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Network Template: Criterion
(defun criterion (criterion X Y &key (reductions nil))
  (apply #'call->
	 (funcall criterion X Y)
	 (map 'list #'asnode reductions)))
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defsequence MLP-Sequence (in-features hidden-dim out-features
			   &key (activation #'!relu))
	     "Three Layer MLP Model"
	     (LinearLayer in-features hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim out-features))

(defun build-mlp-model (&key in-class out-class (hidden-size 256) (activation #'!relu) (lr 1e-3))
  (let* ((mlp (MLP-Sequence in-class hidden-size out-class :activation activation))
	 (lazy-loss (criterion #'softmax-cross-entropy
			       (call mlp
				     (make-input `(batch-size ,in-class) :X))
			       (make-input `(batch-size ,out-class) :Y)
			       :reductions (list #'!sum #'->scal)))
	 (model     (build lazy-loss :inputs `(:X :Y))))
    (mapc (hooker x (Adam x :lr lr)) (model-parameters model))
    (values model mlp)))

(defun step-train-mlp (model x y)
  (let ((act-loss (forward model x y)))
    (backward model)
    (mapc #'call-optimizer! (model-parameters model))
    (/ (tensor-vec act-loss) 100)))

(defmethod accuracy ((model MLP-Sequence) x y)
  (let* ((out   (!argmax (call model x)))
	 (label (!argmax y))
	 (total (proceed (->scal (!sum (A=B out label))))))
    (float (/ (tensor-vec total) (nth 0 (shape out))))))

;; [TODO] Make it simple
;; Changing the visible area without making a copy:
(defun tensor-displace-to (tensor index)
  (setf (tensor-initial-offset tensor) (* index (second (shape tensor))))
  tensor)

 
(defparameter *reshaped-train-img-data* (proceed (!div (!reshape *train-data*  t (* 28 28)) 255.0)))
(defparameter *reshaped-test-img-data* (proceed (!div (!reshape *test-data*   t (* 28 28)) 255.0)))


(defun train-and-valid-mlp (&key (epoch-num 10) (benchmark-p t))
  (multiple-value-bind (compiled-model model)
      (build-mlp-model :in-class 784 :out-class 10 :lr 1e-3)
    (let* ((train-img *reshaped-train-img-data*)
	   (test-img  *reshaped-test-img-data*)
	   (train-img-window (view train-img `(0 100) t))
	   (train-label (view *train-label* `(0 100) t))
	   (test-label *test-label*)
	   (total-loss 0.0))

      (format t "[Log] Start Training...~%")
      (loop for nth-epoch upfrom 1 upto epoch-num do
	(format t "Epoch ~a:~%" nth-epoch)

	(time
	 (loop for batch fixnum upfrom 0 below 60000 by 100 do
	   ;; :X = Train[batch:batch+100, :]
	   (incf total-loss
		 (step-train-mlp compiled-model
				 (tensor-displace-to train-img-window batch)
				 (tensor-displace-to train-label      batch)))))
	(format t "Training Loss: ~a~%~%" (/ total-loss 600))
	(setq total-loss 0.0))

      (format t "Validating...~%")
      (with-no-grad
	(let ((acc (accuracy model test-img test-label)))
	  (format t "Validation Accuracy: ~a~%~%" acc)

	  ;; For github workflow
	  (with-open-file (str "./report.txt"
			       :direction :output
			       :if-exists :supersede
			       :if-does-not-exist :create)
	    ;; 
	    (format str "~a" acc))))

      (when benchmark-p
	(format t "Benchmarking (Forward Step, 1Epoch, n-sample=600)...~%")
	(proceed-bench
	 (!sum (softmax-cross-entropy
		(call
		 (MLP-Sequence 784 256 10)
		 (randn `(100 784)))
		(randn `(100 10))))
	 :n-sample 600
	 :backward t)
	compiled-model))))

;; (cl-waffe2/vm.generic-tensor::reset-compiled-function-cache!)
;; (train-and-valid-mlp :epoch-num 11 :benchmark-p nil)

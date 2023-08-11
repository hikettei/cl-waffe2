
(in-package :cl-waffe2/benchmark)



(defsequence MLP-Sequence (in-features hidden-dim out-features
			   &key (activation #'!relu))
	     "Three Layers MLP Model"
	     (LinearLayer in-features hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim out-features))


(defun build-mlp ()
  (time
   (build
    (!mean
     (softmax-cross-entropy
      (call
       (MLP-Sequence 100 10 5)
       (make-input `(batch-size 100) :X))
      (make-input `(batch-size ,5) :Y))))))



(defun start-benchmark ()
  (start-math-bench)
  (start-composed-bench)
  (start-model-bench))

(start-benchmark)

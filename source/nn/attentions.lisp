
(in-package :cl-waffe2/nn)


(defun sdot-attention (q k v &key (mask nil))
  (let* ((n-dim (car (last (shape q))))
	 (coeff (!sqrt n-dim))
	 (weight (!div (!matmul q (!t k)) coeff)))
    (when mask
      (setq weight (!mul weight mask)))
    (!matmul (!softmax weight) v)))


(defmodel (MultiHeadAttention (self
			       embedding-dim
			       n-heads
			       &aux
			       (head-dim (the fixnum (/ embedding-dim n-heads))))
	   :slots ((l-q :initarg :l-q :reader l-q)
		   (l-k :initarg :l-k :reader l-k)
		   (l-v :initarg :l-v :reader l-v)
		   (l-o :initarg :l-o :reader l-o))
	   :initargs (:l-q (LinearLayer embedding-dim head-dim)
		      :l-k (LinearLayer embedding-dim head-dim)
		      :l-v (LinearLayer embedding-dim head-dim)
		      :l-o (LinearLayer embedding-dim embedding-dim))
	   :on-call-> ((self q k v mask)

		       )))


		   

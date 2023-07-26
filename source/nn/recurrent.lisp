
(in-package :cl-waffe2/nn)

;;
;; TODO:
;; Implement several NN methods to discover the problems/refinements
;;


;; = [RNN] ================================
;;   ...
;;    |
;;    |------|
;; [RNNCell] | Applies RNNCell word by word.
;;    |------|
;;    |
;;   ...
;; ========================================

;;
;; [RNNCell] is a statically working node (JIT is applied), but the part where iterates [RNNCell] are dynamically working depending on the length of words.
;;


(defmodel (RNNCell (self input-size hidden-size
			 &key
			 (activation #'!tanh)
			 (bias t)
			 ;; (dropout nil) ;; TODO: ADD
			 ;;(init-with :uniform) : TODO: Add Orthogonal
			 )
	   :documentation "RNNCell is a..."
	   :slots ((weight :initarg :weight :reader weight-of)
		   (recurrent-weight :initarg :recurrent-weight :reader past-weight-of)
		   (bias1 :initarg :bias1 :reader bias1-of)
		   (bias2 :initarg :bias2 :reader bias2-of)
		   ;; (dropout)
		   (activation :initarg :activation :reader activation-of))
	   
	   :initargs (:weight           (xavier-uniform `(,input-size  ,hidden-size) :requires-grad t)
		      :recurrent-weight (xavier-uniform `(,hidden-size ,hidden-size) :requires-grad t)
		      :bias1 (when bias
			       (uniform-random `(,hidden-size) -0.01 0.01 :requires-grad t))
		      :bias2 (when bias
			       (uniform-random `(,hidden-size) -0.01 0.01 :requires-grad t))
		      :activation activation)
	   
	   :where (X_t[batch-size one input-size] hidden_t[batch-size one hidden-size] -> hidden_t+1[batch-size one hidden-size] where one = 1)
	   :on-call-> ((self x hidden)
		       (let* ((x1 (!matmul x      (!flexible (weight-of self))))
			      (h1 (!matmul hidden (!flexible (past-weight-of self))))
			      (x1 (if (bias1-of self)
				      (!add x1 (%transform (bias1-of self)[hidden-size] -> [~ hidden-size]))
				      x1))
			      (h1 (if (bias2-of self)
				      (!add h1 (%transform (bias2-of self)[hidden-size] -> [~ hidden-size]))
				      h1))
			      (h1 (!add x1 h1))
			      ;; TODO: Dropout
			      (h1 (if (activation-of self)
				      (funcall (activation-of self) h1)
				      h1)))
			 h1))))

#|
I have no idea how to iterate RNNCell...
(defmodel (RNN (self input-size hidden-size
		     &key
		     (n-layers 1)
		     (activation #'!tanh)
		     (bias t)
		     &aux
		     (rnn-layers (loop for i upfrom 0 below n-layers
				       collect (RNNCell input-size hidden-size
							:activation activation
							:bias bias))))
	   :slots ((rnn-layers :initarg :rnn-layers :reader layers-of))
	   :initargs (:rnn-layers rnn-layers)
	   :where (X[batch-size sequence-len dim] Hidden[batch-size one hidden-size] -> Hidden_t+1[batch-size one hidden-size])
	   :on-call-> ((self x hs)
		       (loop for layer in (layers-of self)
			     do (setq hs (call layer x hs)))
		       hs)))

|#

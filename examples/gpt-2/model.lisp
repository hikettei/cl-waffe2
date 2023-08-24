
(in-package :gpt-2-example)

;; In order to measure the performance of Fusion Ops, I've deciced to make a inference on GPT-2 on cl-waffe2 in advance.

;; This file provides an example case of inferecing GPT-2 Model with cl-waffe2

;; [TODO] Python script Converting ONNX model into cl-waffe2 model (make any format? for cl-waffe2?)
;; [TODO] Add: EmbeddingLayer, !split !concatenate?


(defparameter *model-params*
  `((:n-vocab . 50257)
    (:n-ctx   . 1024)
    (:n-emb   . 768)
    (:n-head  . 12)
    (:n-layer . 12)))

(defmacro with-gpt2-config ((&key
			       (n-vocab 50257)
			       (n-ctx 1024)
			       (n-emb 786)
			       (n-head 12)
			       (n-layer 12))
			    &body
			      body)
  `(let ((*model-params*
	   `((:n-vocab . ,,n-vocab)
	     (:n-ctx   . ,,n-ctx)
	     (:n-emb   . ,,n-emb)
	     (:n-head  . ,,n-head)
	     (:n-layer . ,,n-layer))))
     ,@body))

(defun read-config (keyword)
  "(read-config :n-vocab) ;; => 50257"
  (let ((result (find keyword *model-params* :test #'eql :key #'car)))
    (if result
	(cdr result)
	(error "No such a keyword: ~a" keyword))))

(defmodel (GPT2Layer (self save-dir nth-layer)
	   :slots ((ln-1-g)
		   (ln-1-b)
		   
		   (ln-2-g)
		   (ln-2-b)

		   ;; Attention
		   (attn-attn-w)
		   (attn-attn-b)

		   (attn-proj-w)
		   (attn-proj-b)

		   ;; MLP
		   (mlp-fc-w)
		   (mlp-fc-b)

		   (mlp-proj-w)
		   (mlp-proj-b)
		   (nth-layer :initarg :nth-layer))
	   :on-call-> ((self x)
		       (declare (ignore self))
		       x))
  (let* ((layer-dir (format nil "~a/h~a" save-dir nth-layer)))
    ;; layer-dir = save_dir/hN/...
    (setf (slot-value self 'ln-1-g) (load-npy "~a/ln_1/g.npy" layer-dir)
	  (slot-value self 'ln-1-b) (load-npy "~a/ln_1/b.npy" layer-dir)

	  (slot-value self 'ln-2-g) (load-npy "~a/ln_2/g.npy" layer-dir)
	  (slot-value self 'ln-2-b) (load-npy "~a/ln_2/b.npy" layer-dir)

	  (slot-value self 'attn-attn-w) (load-npy "~a/attn/c_attn/w.npy" layer-dir)
	  (slot-value self 'attn-attn-b) (load-npy "~a/attn/c_attn/b.npy" layer-dir)

	  (slot-value self 'attn-proj-w) (load-npy "~a/attn/c_proj/w.npy" layer-dir)
	  (slot-value self 'attn-proj-b) (load-npy "~a/attn/c_proj/b.npy" layer-dir)

	  (slot-value self 'mlp-fc-w) (load-npy "~a/mlp/c_fc/w.npy" layer-dir)
	  (slot-value self 'mlp-fc-b) (load-npy "~a/mlp/c_fc/b.npy" layer-dir)

	  (slot-value self 'mlp-proj-w) (load-npy "~a/mlp/c_proj/w.npy" layer-dir)
	  (slot-value self 'mlp-proj-b) (load-npy "~a/mlp/c_proj/w.npy" layer-dir))))
	  

	  


(defmodel (GPT2 (self &key (save-dir "./examples/gpt-2/assets/models/gpt-2-117M/gpt2-waffe2/model"))
	   :slots ((ln-f-g)
		   (ln-f-b)
		   (wte)
		   (wpe)
		   (layers))
	   :on-call-> ((self x)
		       (declare (ignore self))
		       ;; repeat for nlayer
		       x))
  (let ((n-layer (read-config :n-layer)))
    (setf (slot-value self 'wte) (load-npy "~a/wte.npy" save-dir)
	  (slot-value self 'wpe) (load-npy "~a/wpe.npy" save-dir)
	  (slot-value self 'ln-f-g) (load-npy "~a/ln_f/g.npy" save-dir)
	  (slot-value self 'ln-f-b) (load-npy "~a/ln_f/b.npy" save-dir))
	  
    (setf (slot-value self 'layers)
	  (loop for layer-n upfrom 0 below n-layer
		collect (GPT2Layer save-dir layer-n)))))



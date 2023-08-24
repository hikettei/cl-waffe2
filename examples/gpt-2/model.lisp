
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


;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Model definitions
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
	   :on-call-> gpt2-layer-call)
  (let* ((layer-dir (format nil "~a/h~a" save-dir nth-layer)))
    ;; layer-dir = save_dir/hN/...
    (setf (slot-value self 'ln-1-g)      (load-npy "~a/ln_1/g.npy" layer-dir)
	  (slot-value self 'ln-1-b)      (load-npy "~a/ln_1/b.npy" layer-dir)

	  (slot-value self 'ln-2-g)      (load-npy "~a/ln_2/g.npy" layer-dir)
	  (slot-value self 'ln-2-b)      (load-npy "~a/ln_2/b.npy" layer-dir)

	  (slot-value self 'attn-attn-w) (load-npy "~a/attn/c_attn/w.npy" layer-dir)
	  (slot-value self 'attn-attn-b) (load-npy "~a/attn/c_attn/b.npy" layer-dir)

	  (slot-value self 'attn-proj-w) (load-npy "~a/attn/c_proj/w.npy" layer-dir)
	  (slot-value self 'attn-proj-b) (load-npy "~a/attn/c_proj/b.npy" layer-dir)

	  (slot-value self 'mlp-fc-w)    (load-npy "~a/mlp/c_fc/w.npy" layer-dir)
	  (slot-value self 'mlp-fc-b)    (load-npy "~a/mlp/c_fc/b.npy" layer-dir)

	  (slot-value self 'mlp-proj-w)  (load-npy "~a/mlp/c_proj/w.npy" layer-dir)
	  (slot-value self 'mlp-proj-b)  (load-npy "~a/mlp/c_proj/w.npy" layer-dir))))

;; Custom printings
(defmethod on-print-object ((model GPT2Layer) stream)
  (format stream "~%N_LAYER=~a" (slot-value model 'nth-layer)))	  

;; Forward process of gpt2-layer
(defmethod gpt2-layer-call ((self GPT2Layer) x y)

  )

(defmodel (GPT2 (self &key (save-dir "./examples/gpt-2/assets/models/gpt-2-117M/gpt2-waffe2/model"))
	   :slots ((ln-f-g)
		   (ln-f-b)
		   (wte)
		   (wpe)
		   (layers))
	   :where (X-out[~ sentence-length embedding-size] Source[~ sentence-length] Target[~ sentence-length embedding-size]
			   ->
			   X-out[~ sentence-length embedding-size]
			   where
			   embedding-size = (read-config ':n-emb))
	   :on-call-> gpt2-call)
  (let ((n-layer (read-config :n-layer)))
    (setf (slot-value self 'wte)    (load-npy "~a/wte.npy" save-dir)
	  (slot-value self 'wpe)    (load-npy "~a/wpe.npy" save-dir)
	  (slot-value self 'ln-f-g) (load-npy "~a/ln_f/g.npy" save-dir)
	  (slot-value self 'ln-f-b) (load-npy "~a/ln_f/b.npy" save-dir))
	  
    (setf (slot-value self 'layers)
	  (loop for layer-n upfrom 0 below n-layer
		collect (GPT2Layer save-dir layer-n)))))

;; Customized printings
(defmethod on-print-object ((model GPT2) stream)
  (format stream "~%  [Layers]:~%~a~%"
	  (with-output-to-string (out)
	    (dolist (layer (slot-value model 'layers))
	      (format out "~a~%" layer)))))

;; Forward process for GPT2
(defmethod gpt2-call ((self GPT2) x-out x y)
  (with-slots ((wte wte) (wpe wpe) (layers layers) (ln-f-g ln-f-g) (ln-f-b ln-f-b)) self
    
    (call-> x-out
	    (asnode #'!gpt2-load-pe x wte wpe) ;; X-out <- GPT2Pe(x, wte, wpe)
	    )))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Inference/Exports/Tokenizers
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


(defun encode-sentence ())
(defun decode-sentence ())

;; [TODO] proceed-bench
(defun build-gpt2 ()
  (with-devices (CPUTensor LispTensor)
    (with-no-grad
      (build
       (!argmax
	(call (GPT2)
	      (make-input `(1 sentence-length) :input-sentence)
	      (make-input `(1 sentence-length) :memory)))))))

(defun inference-gpt2 (model sentence)
  (with-memory-pool ;; Outside of this block, gc is called.
    (set-input model :input-sentence sentence)
    (let ((tokens (forward model)))
      ;; Tokens
      )))

;; Invokes REPL form
(defun launch-repl ()
  (let ((model (GPT2)))
    (print "Loaded!")
    (print model)))



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
			       (n-emb 768)
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
  (let ((keyword (if (keywordp keyword)
		     keyword
		     (intern (format nil "~a" keyword) "KEYWORD"))))
    (let ((result (find keyword *model-params* :test #'eql :key #'car)))
      (if result
	  (cdr result)
	  (error "No such a keyword: ~a" keyword)))))


;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Model definitions
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defmodel (GPT2Layer (self orig save-dir nth-layer)
	   :slots ((orig :initarg :orig :initform nil) ;; The GPT2 Model it belonging to
		   (ln-1-g :initform nil)
		   (ln-1-b :initform nil)
		   
		   (ln-2-g :initform nil)
		   (ln-2-b :initform nil)

		   ;; Attention
		   (attn-attn-w :initform nil)
		   (attn-attn-b :initform nil)

		   (attn-proj-w :initform nil)
		   (attn-proj-b :initform nil)

		   ;; MLP
		   (mlp-fc-w :initform nil)
		   (mlp-fc-b :initform nil)

		   (mlp-proj-w :initform nil)
		   (mlp-proj-b :initform nil)
		   (nth-layer :initarg :nth-layer :initform nil))
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
(defmethod gpt2-layer-call ((self GPT2Layer) x)
  (with-slots ((orig orig)
	       (ln-1-g ln-1-g)
	       (ln-1-b ln-1-b)
	       (ln-2-g ln-2-g)
	       (ln-2-b ln-2-b)
	       (mlp-fc-w mlp-fc-w)
	       (mlp-fc-b mlp-fc-b)
	       (mlp-proj-w mlp-proj-w)
	       (mlp-proj-b mlp-proj-b)
	       (attn-attn-w attn-attn-w)
	       (attn-attn-b attn-attn-b)
	       (attn-proj-w attn-proj-w)
	       (attn-proj-b attn-attn-b))
      self

    (let* ((xp
	     (call-> x
		     (asnode #'!gpt2-layernorm ln-1-g ln-1-b)
		     (asnode #'!affine attn-attn-w attn-attn-b) ;; X[N Length Embedding_Dim] @ W[1 786 2304] + B[2304]
		     (asnode #'self-attention orig)
		     (asnode #'!affine attn-proj-w attn-proj-b)))
	   (proj-x
	     (call-> x
		     (asnode #'!add xp) ;; Residual Connection
		     ;; Feed Forward Network
		     (asnode #'!gpt2-layernorm ln-2-g ln-2-b)
		     (asnode #'!affine mlp-fc-w mlp-fc-b) ;; X(3072 N).T @ W(1 768 3072) + B(3072)
		     (asnode #'!gelu)
		     (asnode #'!affine mlp-proj-w mlp-proj-b))))
      (!add xp proj-x))))


(defmodel (GPT2 (self &key (save-dir "./examples/gpt-2/assets/models/gpt-2-117M/gpt2-waffe2/model"))
	   :slots ((ln-f-g :initform nil)
		   (ln-f-b :initform nil)
		   (wte    :initform nil)
		   (wpe    :initform nil)
		   (layers :initform nil)

		   (memory-k :initform nil)
		   (memory-v :initform nil))
	   :on-call-> gpt2-call)
  (let ((n-layer (read-config :n-layer)))
    (setf (slot-value self 'wte)    (load-npy "~a/wte.npy" save-dir)
	  (slot-value self 'wpe)    (load-npy "~a/wpe.npy" save-dir)
	  (slot-value self 'ln-f-g) (load-npy "~a/ln_f/g.npy" save-dir)
	  (slot-value self 'ln-f-b) (load-npy "~a/ln_f/b.npy" save-dir))
	  
    (setf (slot-value self 'layers)
	  (loop for layer-n upfrom 0 below n-layer
		collect (GPT2Layer self save-dir layer-n)))))

;; Customized printings
(defmethod on-print-object ((model GPT2) stream)
  (format stream "~%  [Layers]:~%~a~%"
	  (with-output-to-string (out)
	    (dolist (layer (slot-value model 'layers))
	      (format out "~a~%" layer)))))

(defun !gpt2-layers (x-out self)
  (with-slots ((layers layers)) self
    (loop for layer in layers do
      (setq x-out (call layer x-out)))
    x-out))

;; Forward process for GPT2
(defmethod gpt2-call ((self GPT2) x-out x)
  (with-slots ((wte wte) (wpe wpe) (ln-f-g ln-f-g) (ln-f-b ln-f-b)) self
    
    (call-> x-out
	    (asnode #'!gpt2-load-pe   x wte wpe) ;; X-out <- GPT2Pe(x, wte, wpe)
	    (asnode #'!gpt2-layers    self)
	    (asnode #'!gpt2-layernorm ln-f-g ln-f-b))))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Inference/Exports/Tokenizers
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


(defun encode-sentence ())
(defun decode-sentence ())

(defun extend-source-input (model source N)
  ;; Extend the source into N+1 Area
  (set-input model
	     (let ((out (!move (!view
				(make-tensor `(,(car (shape source)) ,(1+ N) ,(third (shape source))))
				t
				`(0 ,N))
			       source)))
	       (proceed (!view out t t t)))
	     :x-source))

;; Invokes REPL form
(defun launch-repl (&key (use-model nil))
  (with-no-grad
    (let ((model (or use-model (GPT2))))
      (print "Loaded!")
      (print model)
      (print
       (proceed-bench
	(call model
	      (ax+b `(1 100 768) 0 0) ;;(make-input `(1 N 768) :x-source) ;; X_Source
	      (ax+b `(12 100) 0 1))))          ;; Input Words

      (let* ((source (make-input `(1 N 769) :x-source))
	     (compiled-model (build (call model
					  source
					  (ax+b `(12 100) 0 1)))))

	(loop for n fixnum upfrom 1 below 12 do
	  (extend-source-input compiled-model source N)
	  (setq source (forward model)))
	source ;; <- decode the source
	))))



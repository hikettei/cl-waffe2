
(in-package :gpt-2-example)

;; 
;; [TODO] Opt: Compiling !matmul
;;        Use: Standard APIs

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
	   :on-call-> gpt2layer-call)
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
	  (slot-value self 'mlp-proj-b)  (load-npy "~a/mlp/c_proj/b.npy" layer-dir))))

;; Custom printings
(defmethod on-print-object ((model GPT2Layer) stream)
  (format stream "~%N_LAYER=~a" (slot-value model 'nth-layer)))	  

;; Forward process of gpt2-layer
(defmethod gpt2layer-call ((self GPT2Layer) x past)
  (declare (type AbstractTensor x)
	   (type (or null list) past))
  (with-slots ((ln-1-g ln-1-g)
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
	       (attn-proj-b attn-proj-b))
      self

    ;; GPT2Layer = Block(LayerNorm, Attention, LayerNorm, MLP)
    (let* ((present nil)
	   (attn
	     (call-> x
		     (asnode #'LayerNorm-Revisit ln-1-g ln-1-b)
		     ;; Projection: 786 -> 786*3		    
		     (asnode #'!matmul attn-attn-w) ;; X[Batch N Embedding_Dim] @ W[786 2304] + B[2304]
		     (asnode #'!add (%transform attn-attn-b[i] -> [~ i]))
		     (assetq (nil present) #'SelfAttention past) ;; NIL, PRESENT <- SelfAttention(x, past)
		     (asnode #'!matmul attn-proj-w)
		     (asnode #'!add (%transform attn-proj-b[i] -> [~ i]))))
	   (x (!add x attn)) ;; Residual Connection
	   (m
	     (call-> x
		     ;; Feed Forward Network
		     (asnode #'LayerNorm-Revisit ln-2-g ln-2-b)
		     (asnode #'!matmul mlp-fc-w) ;; X(768 N).T @ W(1 768 3072) + B(3072)
		     (asnode #'!add    (%transform mlp-fc-b[i]   -> [~ i]))
		     (asnode #'!gelu-lisptanh)
		     (asnode #'!matmul mlp-proj-w)
		     (asnode #'!add    (%transform mlp-proj-b[i] -> [~ i])))))
      ;; Residual Connection
      (values (!add x m) present))))


(defmodel (GPT2 (self &key (save-dir "./examples/gpt-2/assets/models/gpt-2-117M/gpt2-waffe2/model"))
	   :slots ((ln        :initform nil)
		   (embedding :initform nil)
		   (wte       :initform nil)
		   (wpe       :initform nil)
		   (layers    :initform nil)))
  (let ((n-layer (read-config :n-layer)))    
    (setf (slot-value self 'embedding) (GPT2PositionalEmbedding
					(read-config :n-vocab)
					(read-config :n-ctx)
					(read-config :n-emb))
	  (slot-value self 'wte)    (load-npy "~a/wte.npy" save-dir)
	  (slot-value self 'wpe)    (load-npy "~a/wpe.npy" save-dir))

    (let* ((alpha (load-npy "~a/ln_f/g.npy" save-dir))
	   (beta  (load-npy "~a/ln_f/b.npy" save-dir)))
      ;; Initializing alpha beta when creating LayerNorm
      ;; Is nothing but waste of memory??
      (setf (slot-value self 'ln) (LayerNorm (shape alpha))
	    (alpha-of (slot-value self 'ln)) alpha
	    (beta-of (slot-value self 'ln)) beta))    
    
    (setf (slot-value self 'layers)
	  (loop for layer-n upfrom 0 below n-layer
		collect (GPT2Layer self save-dir layer-n)))))

(defmethod call ((self GPT2) &rest inputs)
  ;; Inputs: Prev Past1 Past2 Past3 ...
  (let ((prev        (car inputs))
	(layer-pasts (cdr inputs))
	(presents nil))
    (with-slots ((layers layers) (embedding embedding) (wte wte) (wpe wpe) (ln ln)) self
      ;; Return: (values output presents)
      (values
       (call->
	(make-input `(,(car (shape prev)) ,(second (shape prev)) ,(read-config :n-emb)) nil)
	
	;; Composes: [PE] -> [N * Layers] -> LayerNorm
	(asnode #'(lambda (x) (call embedding prev wte wpe x)))
	(asnode
	 #'(lambda (x-out &aux (present nil))
	     (loop for layer in layers
		   for n upfrom 0 do
		     (multiple-value-setq (x-out present)
		       (call layer x-out (nth n layer-pasts)))
		     (push present presents))
	     x-out))
	ln)
       presents))))

;; Customized printings
(defmethod on-print-object ((model GPT2) stream)
  (format stream "~%  [Layers]:~%~a~%"
	  (with-output-to-string (out)
	    (dolist (layer (slot-value model 'layers))
	      (format out "~a~%" layer)))))

(defmethod lm-head ((self GPT2) x)
  (!matmul (!rankup x -1) (!t (slot-value self 'wte))))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Inference/Exports
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun compile-gpt2-model (model &key (disassemble nil) (bench nil))
  (with-no-grad
    (let* ((compiled-model (build (call model (make-input `(batch-size N) :prev)) :inputs `(:prev))))
      (when disassemble
	(disassemble-waffe2-ir (call model (make-input `(batch-size N) :prev))))

      (when bench
	(proceed-bench
	 (call model (ax+b `(1 10) 0 1))))
      
      compiled-model)))

(defun start-token () (gethash "<|endoftext|>" *encoder-json*))

(defun gpt2-inference (model compiled-model source &key (length 10) (temperature 1.0))
  (declare (ignore temperature))

  (let ((decode-list))
    (dotimes (i length)
      (format t "[~a/~a]~%" i length)
      (let ((result (proceed (->scal (!argmax (lm-head model (!view (forward compiled-model source) t -1))))))
	    (new-array (ax+b `(,(car (shape source)) ,(1+ (second (shape source)))) 0 0)))
	(push (tensor-vec result) decode-list)
	(with-facets ((s* (source :direction 'simple-array))
		      (a* (new-array :direction 'simple-array)))
	  (loop with s* list = (coerce s* 'list)
		for idx upfrom 0 below (second (shape new-array)) do
		  (setf (aref a* idx) (or (nth idx s*)
					  (+ 0.0 (tensor-vec result))))))
	(setq source new-array)))
    (reverse decode-list)))


;; Workload:
;; 1. inference anyway
;; 2. do a cache
;; Invokes REPL form

;; It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him.

(defun launch-repl (&key (use-model nil) (length 50) (temperature 1.0))
  (format t "length=~a~%" length)
  
  (with-no-grad
    (let ((model (or use-model (GPT2))))
      (format t "[INFO] The model was restored from the trained weight!~%")
      (print model)
      (when (null *encoder-json*)
	(format t "[INFO] Loading encoder...~%")
	(load-bpe-merges)
	(load-encoder-json))

      (loop named repl while t do
	(format t "~%Type \"quit\" to exit, \"benchmark\" to start profiling.~%>Type anything to start generating a sentence.~%Note that GPT2 Inference is stil unstable...~%")
	(let ((input (read-line)))
	  
	  (when (equal input "quit")
	    (format t "Good bye. You can use (gpt-2-example:launch-repl) to invoke me again. ~%")
	    (return-from repl))

	  (format t "[INFO] Compiling GPT2 Model...~%")
	  
	  (let* ((compiled-model (time (compile-gpt2-model model))))	    
	    (if (equal input "benchmark")
		(progn
		  (format t "N_SAMPLE=10, LENGTH=10~%")
		  (proceed-bench
		   (call model (ax+b `(1 10) 0 0))
		   :n-sample 10))
		(let* ((source (encode-sentence input)))
		  (let ((generated-sentence-list (gpt2-inference model compiled-model source :length length :temperature temperature)))
		    (format t "~%GPT2> ~a~%" (decode-sentence generated-sentence-list)))
		  
		  (return-from launch-repl)))))))))



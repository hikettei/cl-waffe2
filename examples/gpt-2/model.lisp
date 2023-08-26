
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
	  (slot-value self 'mlp-proj-b)  (load-npy "~a/mlp/c_proj/b.npy" layer-dir))))

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
	       (attn-proj-b attn-proj-b))
      self

    (let* ((attn
	     (call-> x
		     (asnode #'!gpt2-layernorm ln-1-g ln-1-b)
		     ;; Projection: 786 -> 786*3
		     (asnode #'!matmul attn-attn-w) ;; X[Batch N Embedding_Dim] @ W[786 2304] + B[2304]
		     (asnode #'!add (%transform attn-attn-b[i] -> [~ i]))
		     (asnode #'self-attention orig)
		     (asnode #'!matmul attn-proj-w)
		     (asnode #'!add (%transform attn-proj-b[i] -> [~ i]))))
	   (x (!add x attn)) ;; Residual Connection
	   (m
	     (call-> x
		     ;; Feed Forward Network
		     (asnode #'!gpt2-layernorm ln-2-g ln-2-b)
		     (asnode #'!matmul mlp-fc-w) ;; X(768 N).T @ W(1 768 3072) + B(3072)
		     (asnode #'!add    (%transform mlp-fc-b[i]   -> [~ i]))
		     (asnode #'!gelu-lisptanh)
		     (asnode #'!matmul mlp-proj-w)
		     (asnode #'!add    (%transform mlp-proj-b[i] -> [~ i])))))
      ;; Residual Connection
      (!add x m))))


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

(defmethod lm-head ((self GPT2) x)
  (!matmul (!rankup x -1) (!t (slot-value self 'wte))))


;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Tokenizers
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; Reference: https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/encoder.py

(defparameter *encoder-json* nil)
(defparameter *decoder-json* nil)

(defparameter *pat* (create-scanner "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"))

(defparameter *bpe-merges* nil)

(defun load-bpe-merges (&key (save-path "./examples/gpt-2/assets/models/gpt-2-117M/vocab.bpe"))
  (let* ((bpe (uiop:read-file-string save-path))
	 (bpe (subseq bpe 1 (1- (length bpe))))
	 (out (make-hash-table :test #'equal))
	 (pairs (cdr (loop for mstr in (split "\\n" bpe) collect (split " " mstr)))))
    (loop for p in pairs
	  for i upfrom 0 do
	    (setf (gethash p out) i))
    (setf *bpe-merges* out)
    t))

(defun load-encoder-json (&key (save-path "./examples/gpt-2/assets/models/gpt-2-117M/encoder.json"))
  (format t "[INFO] Loading encoder.json ...~%")
  (let ((encoder-str (time (parse (uiop:read-file-string save-path))))
	(dict (make-hash-table :test #'equal))
	(dec-dict (make-hash-table)))
    (format t "[INFO] Parsing was done... n_vocab=~a~%" (/ (length encoder-str) 2))
    (loop while encoder-str do
      (let ((key (pop encoder-str))
	    (val (pop encoder-str)))
	(setf (gethash val dec-dict) (format nil "~a" key))
	(setf (gethash (format nil "~a" key) dict) val)))
    (setf *decoder-json* dec-dict)
    (setf *encoder-json* dict)))

(defun get-pairs (token)
  (declare (type string token))

  ;; token ... Hi Gthere, ...
  (loop for index fixnum upfrom 0 below (1- (length token))
	collect
	(list (string (aref token index)) (string (aref token (1+ index))))))

(defun countup-nth (word token n)
  (let ((count 0)
	(n (1+ n)))
    (loop for tkn in token
	  for pos upfrom 0
	  if (equal tkn word)
	    do (incf count 1)
	  if (= count n)
	    do (return-from countup-nth pos))))

(defun bpe-split (token)
  (declare (type string token))
  (let ((word (list token))
	(out-of-range (* -1 (+ 1 (length (hash-table-keys *bpe-merges*)))))
	(pairs (get-pairs token)))

    (loop named bpe-iter while t do
      (let* ((smallest (loop for pair in pairs minimize (or (gethash pair *bpe-merges*) out-of-range)))
	     (bigram   (find smallest pairs :test #'eql :key #'(lambda (x) (gethash x *bpe-merges*)))))
	(when (null bigram)
	  (return-from bpe-iter))

	(multiple-value-bind (first second) (apply #'values bigram)
	  (let ((new-word)
		(i 0))
	    (loop named bpe-word-iter while (< i (length word)) do
	      (if (or (null (find first word :test #'equal))
		      (not (< i (count first word :test #'equal))))
		  (progn
		    ;; Break
		    (setq new-word
			  `(,@new-word
			    ,@(subseq word i (length word))))
		    (return-from bpe-word-iter))
		  (let ((j (countup-nth first word i)))
		    (setq new-word
			  `(,@new-word
			    ,@(subseq word i j)))
		    (setq i j)
		    (if (and (equal (nth i word) first)
			     (< i (1- (length word)))
			     (equal (nth (1+ i) word) second))
			(progn
			  (setq new-word
				`(,@new-word
				  ,(concatenate 'string first second)))
			  (incf i 2))
			(progn
			  (setq new-word
				`(,@new-word ,(nth i word)))
			  (incf i 1))))))
	    (setq word new-word)
	    (if (= (length word) 1)
		(return-from bpe-iter)
		(setq pairs (get-pairs word)))))))
    word))


(defun encode-sentence (sentence) ;; (read-line)
  (declare (type string sentence))
  (let ((tokens (all-matches-as-strings *pat* sentence))
	(bpe-tokens))
    (loop for token in tokens do
      (let* ((token (loop for n upfrom 0 below (length token)
			  collect (gethash (char-code (aref token n)) *byte2unicode*)))
	     (token (apply #'concatenate 'string token)))
	(dolist (bpetoken (bpe-split token))
	  (push (+ 0.0 (or (gethash bpetoken *encoder-json*) 0)) bpe-tokens))))
    (let ((tokens (reverse bpe-tokens)))
      (change-facet
       (make-array `(1 ,(length tokens))
		   :element-type 'single-float
		   :initial-contents `(,tokens))
       :direction 'AbstractTensor))))

(defun decode-sentence (list)
  (declare (type list list))
  (let ((text (apply #'concatenate 'string (loop for token in list collect (gethash token *decoder-json*)))))
    (with-output-to-string (out)
      (loop for pos fixnum upfrom 0 below (length text) do
	(let ((code (gethash (char-code (aref text pos)) *byte2unicode*)))
	  (if code
	      (princ code out)
	      (princ " " out)))))))


;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Inference/Exports
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defparameter N 0 "Indicates the length of sentence")

;; [TODO] Optimize it:
;; [TODO] Impl: !concat
(defun extend-source-input (model name source N)
  ;; Extend the source into N+1 Area
  (set-input model
	     name
	     (let ((out (!move (!view
				(make-tensor `(,(car (shape source)) ,(+ 1 N) ,(third (shape source))))		
				t
				`(0 ,N))
			       source)))
	       (proceed (->contiguous (!view out t `(0 ,(+ 1 N))))))))

(defun extend-source-input-2d (model name source N extend-with)
  ;; Extend the source into N+1 Area
  (set-input model
	     name
	     (let ((out (!move (!view
				(make-tensor `(,(car (shape source)) ,(+ 1 N)))
				t
				`(0 ,N))
			       source)))
	       (proceed (->contiguous (!view out t `(0 ,(+ 1 N)))))))
  (dotimes (batch-n (car (shape source)))
    (setf (mref (get-input model name) batch-n N) extend-with)))

(defun start-token () (gethash "<|endoftext|>" *encoder-json*))

(defun gpt2-inference (model compiled-model source input &key (length 10) (temperature 1.0))
  (declare (ignore temperature))
  ;;mem-k mem-v: Not used for a now
  (setf (slot-value model 'memory-k) nil
        (slot-value model 'memory-v) nil)

  (let ((result))
    (loop with slen fixnum   = (second (shape input))
	  with batch-size    = (car    (shape source))
	  with embedding-dim = (third  (shape source))
	  for nth fixnum upfrom slen below (+ slen length) do
	    (format t "~a/~a...~%" nth (+ slen length))
	    (setq source (get-input compiled-model :x-source))
	    (setq input  (get-input compiled-model :x-input))
	    ;;(print "INPUT")
	    ;;(print (tensor-vec input))
	    (let* ((N (second (shape source))))
	      (let* ((out     (forward compiled-model))
		     (tmp     (make-input `(1 ,N ,(third (shape out))) nil))
		     (tmp     (->contiguous (!view (!move tmp out) 0 -1)))
		     (out     (lm-head model tmp))
		     (idx     (tensor-vec (proceed (->scal (!argmax (!softmax out) :axis 1))))))

		(set-input compiled-model :x-source (make-tensor `(,(car (shape source)) ,(1+ N) ,(third (shape source)))))
		(extend-source-input-2d compiled-model :x-input  input  nth (coerce idx 'single-float))
		(push idx result))))
    (reverse result)))

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
	  
	  (let* ((source         (make-input `(1 N 768) :x-source))  ;; (Batch_Size Sentence_Length Embedding_dim)
		 (input-tensor   (make-input `(1 N)     :x-input))   ;; (Batch_Size Sentence_Length)
		 (compiled-model (time (build (call model source input-tensor)))))
	    
	    (if (equal input "benchmark")
		(progn
		  (format t "N_SAMPLE=10, LENGTH=10~%")
		  (proceed-bench
		   (call model (ax+b `(1 10 768) 0 0) (uniform-random `(1 10) 0 100))
		   :n-sample 10))
		(let* ((input-sentence (encode-sentence input))
		       (initial-length (second (shape input-sentence)))
		       (input-source   (ax+b `(1 ,initial-length 768) 0 0)))
		  
		  ;; X Embedding ... (1 N 768)
		  ;; X Sparse    ... (1 N)

		  (set-input compiled-model :x-source input-source)
		  (set-input compiled-model :x-input  input-sentence)
		  
		  (let ((generated-sentence-list (gpt2-inference model compiled-model input-source input-sentence :length length :temperature temperature)))
		    (format t "~%GPT2> ~a~%" (decode-sentence generated-sentence-list)))
		  
		  (return-from launch-repl)))))))))



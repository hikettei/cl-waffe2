
(in-package :gpt-2-example)

;; Utils

(defun load-npy (path &rest args)
  ;; npz -> AbstractTensor
  (format t "[INFO] load-npy attempts to load ~a...~%" (apply #'format nil path args))
  (parameter (change-facet (numpy-file-format:load-array (apply #'format nil path args)) :direction 'AbstractTensor)))

(defun incf-tensor-ptr (tensor tensor-ptr &key (offset 0))
  (cl-waffe2/backends.cpu::incf-tensor-ptr tensor tensor-ptr :offset offset))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  [AbstractNode] GPT2PositionalEmbedding
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(defun expand-embedding-form (ctx wte wpe wte-ptr wpe-ptr ctx-out-ptr ctx-out ctx-view ctx-out-view)
  ;; Returns a S-expression later compiled
  ;; CTX = [~ sentence-length], Sparse Matrix
  ;; WTE = [Vocab-Size Embedding-Size]
  ;; WPE = [N-CTX Embedding-Size]
  (declare (type AbstractTensor ctx wte wpe ctx-out))

  (let ((embedding-size (second (shape wte))))
    (with-gensyms (position-n vocab-index wte-position wpe-position)
      `(progn
	 (cl-waffe2/backends.cpu::waffe2-smul-scal
	  ,embedding-size
	  (incf-tensor-ptr ,ctx-out ,ctx-out-ptr :offset ,(offset-of ctx-out-view 0))
	  1
	  0.0)
	 (loop for ,position-n fixnum upfrom 0 below ,(second (shape ctx)) do
	   (let* ((,vocab-index (aref (tensor-vec ,ctx) (+ ,position-n ,(offset-of ctx-view 0))))
		  (,wte-position (* (round (the single-float ,vocab-index)) ,embedding-size))
		  (,wpe-position (* ,position-n ,embedding-size)))
	     ;; ctx-out <- add(WTE[Word_Index, :], WPE[Position, :])

	     ;; [TODO] Fuse these steps to get more speed:
	     ;; Using SIMD Extention to add two vectors
	     ;; sadd: Y += X

	     ;; Y *= 0
	     ;; Y += WTE
	     ;; Y += WPE
	     (cl-waffe2/backends.cpu::waffe2-sadd
	      ,embedding-size	      
	      (incf-tensor-ptr ,ctx-out ,ctx-out-ptr :offset (+ ,(offset-of ctx-out-view 0)
								(* ,position-n ,embedding-size))) ;; CTX-OUT[:, pos, embedding-size]
	      1
	      (incf-tensor-ptr ,wte ,wte-ptr :offset ,wte-position)
	      1)

	     (cl-waffe2/backends.cpu::waffe2-sadd
	      ,embedding-size
	      (incf-tensor-ptr ,ctx-out ,ctx-out-ptr :offset (+ ,(offset-of ctx-out-view 0)
								(* ,position-n ,embedding-size))) ;; CTX-OUT[:, pos, embedding-size]
	      1
	      (incf-tensor-ptr ,wpe ,wpe-ptr :offset ,wpe-position)
	      1)))))))

;; Implementing Embedding
;; [TODO] Backward, Include this to :cl-waffe2/nn package
(defnode (GPT2PositionalEmbedding (self vocab-size n-ctx embedding-size)
	  :documentation "add(WTE[CTX], WPE[CTX]) -> CTX"
	  :where (ctx[N sentence-length] wte[vocab-size embedding-size] wpe[n-ctx embedding-size] ctx-out[N sentence-length embedding-size]
			->
			ctx-out[N sentence-length embedding-size])))

(define-impl (GPT2PositionalEmbedding :device LispTensor :cache-when-compiled nil)
	     :forward ((self ctx wte wpe ctx-out)

		       ;; This is intended because I want to explict the reason of error
		       (when (cl-waffe2/backends.cpu::simd-extension-p)
			 (error "GPT2 Example seems working without SIMD-Extension. Because GPT2PositionalEmbedding depends on foreign simd library, SIMD-Extension must be loaded in advance.

You can simply run:
    $ make build_simd_extension

In your terminal, and cl-waffe2 will load it."))

		       (assert
			(and (eql (order ctx) :column)
			     (eql (order wpe) :column)
			     (eql (order wte) :column))
			nil
			"GPT2PositionalEmbedding: Orders must be :column (C Order), not a :row (Fortran Order)")

		       (with-gensyms (wte-ptr wpe-ptr ctx-out-ptr)
			 `(locally (declare (optimize (speed 1)))
			    (cl-waffe2/backends.cpu::with-tensor-ptrs ((,wpe-ptr ,wpe)
								       (,wte-ptr ,wte)
								       (,ctx-out-ptr ,ctx-out))
			      (,@(call-with-view
				  #'(lambda (ctx-view ctx-out-view)
				      (expand-embedding-form ctx wte wpe wte-ptr wpe-ptr ctx-out-ptr ctx-out ctx-view ctx-out-view))
				  (list ctx ctx-out)
				  :at-least-dim 1
				  :force-order t
				  :lparallel nil
				  :fuse nil)
			       ,ctx-out))))))

(defun !gpt2-load-pe (ctx-out ctx wte wpe)
  (call
   (GPT2PositionalEmbedding
    (read-config :n-vocab)
    (read-config :n-ctx)
    (read-config :n-emb))
   ctx wte wpe ctx-out))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Defines Activations
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; [TODO] Move this into :cl-waffe2/nn package
;; eps=1.0e-5, [FixME] rounding error of mean
(defun !gpt2-layernorm (x g b &key (eps 1.0e-5))
  ;; X ... [N, Sentene_length, Embedding_DIM]
  ;; g/b... [Embedding_DIM]

  ;; (!sub x u) != 0.0 ...
  (let* ((u (!mean x :axis -1 :keepdims t))  ;; μ=mean(x, axis=-1)
	 (s (!mean (!expt (!sub x u) 2) :axis -1 :keepdims t))
	 (x (!div (!sub x u)
		  (!sqrt (!add (->contiguous s) eps)))))
    (!add
     (!mul
      x
      (%transform g[i] -> g[~ i]))
     (%transform b[i] -> b[~ i]))))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Defines more utils
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun !affine (x weight bias)
  (!add (!matmul (!t x) (!flexible weight))
	(%transform bias[i] -> bias[~ i])))


;; GeLU but tanh is safe
(defun !gelu-lisptanh (x)
  (!* 0.5 x
      (!+ 1
	  (let ((o (!* (coerce (sqrt (/ 2.0 pi)) (dtype->lisp-type (dtype x)))
		       (!+ x
			   (!* 0.044715 (!expt x 3))))))
	    (with-devices (LispTensor)
	      (!tanh o))))))


;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  MHA
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun split-heads (x qkv &aux (stride (/ (car (last (shape x))) 3)))
  (declare (type (member :Q :K :V) qkv))
  ;; X ... [~ Sentence_length 2304] -> Q[~ Sentece-Length 786] K V...
  (let* ((x (!view x t t (case qkv
			   (:Q `(0 ,stride))
			   (:K `(,stride ,(* 2 stride)))
			   (:V `(,(* 2 stride) ,(* 3 stride))))))
	 (new-x-shape `(,@(butlast (shape x)) ,(read-config :n-head) ,(/ (car (last (shape x))) (read-config :n-head))))
	 (out          (apply #'!reshape x new-x-shape)))
    (if (eql qkv :K)
	(!permute out (torch-order 0 2 3 1)) ;; (batch head head-features N)
	(!permute out (torch-order 0 2 1 3))))) ;; (batch head-f N head)

(defun merge-heads (x)
  (let* ((x (!permute x (torch-order 0 2 1 3)))
	 (x-shape `(,@(butlast (shape x) 2) ,(apply #'* (last (shape x) 2)))))
    (apply #'!reshape x x-shape)))

(defun self-attention (x gpt2) ;; X ...[N 2304]
  (with-slots ((memory-k memory-k) (memory-v memory-v)) gpt2
    (let* ((K (split-heads x :K))
	   (V (split-heads x :V))
	   ;;(x (with-instant-kernel x ;; Embedding Lisp Code Directly to cl-waffe2 IR
	   ;;	`(if (or (slot-value ,gpt2 'memory-k) (slot-value ,gpt2 'memory-v))
	   ;;	     ;; Layer-Past isn't none
	   ;;	     (progn
	   ;;	       (setf (detach-p ,K) T)
	   ;;	       (setf (detach-p ,V) T)
	   ;;	       ;; ... Concatenate past keys
	   ;;	       )
	   ;;	     ,x)))
	   (Q (split-heads x :Q)))

      (let ((w (!matmul Q K)))
	;; (1 768 768)
	(call-> w
		(asnode #'!div (make-tensor (car (last (shape w)))))
		(asnode #'!softmax :avoid-overflow nil)
		(asnode #'!matmul v)
		(asnode #'merge-heads))))))


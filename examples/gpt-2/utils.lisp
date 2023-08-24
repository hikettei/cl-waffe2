
(in-package :gpt-2-example)

;; Utils

(defun load-npy (path &rest args)
  ;; npz -> AbstractTensor
  (format t "[INFO] load-npy attempts to load ~a...~%" (apply #'format nil path args))
  (parameter (change-facet (numpy-file-format:load-array (apply #'format nil path args)) :direction 'AbstractTensor)))

(defun expand-embedding-form (ctx wte wpe ctx-out ctx-view ctx-out-view)
  ;; Returns a S-expression later compiled
  ;; CTX = [~ sentence-length], Sparse Matrix
  ;; WTE = [Vocab-Size Embedding-Size]
  ;; WPE = [N-CTX Embedding-Size]
  (declare (type AbstractTensor ctx wte wpe ctx-out-view))

  (let ((embedding-size (second (shape wte))))
    (with-gensyms (position-n vocab-index wte-position)
      `(loop for ,position-n fixnum upfrom 0 below ,(second (shape ctx)) do
	(let* ((,vocab-index (aref (tensor-vec ,ctx) (+ ,position-n (offset-of ,ctx-view 0))))
	       (,wte-position (* (round (the single-float ,vocab-index)) ,embedding-size)))

	  )))))

;; Implementing Embedding
(defnode (GPT2PositionalEmbedding (self vocab-size n-ctx embedding-size)
	  :documentation "add(WTE[CTX], WPE[CTX]) -> CTX"
	  :where (ctx[~ sentence-length] wte[vocab-size embedding-size] wpe[n-ctx embedding-size] ctx-out[~ sentence-length embedding-size]
			->
			ctx-out[~ sentence-length embedding-size])))

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
		       
		       `(with-ranked-loop ((#'expand-embedding-form ,ctx ,ctx-out)
					   :kernel-size 1
					   :shuffle-rank nil
					   :lparallel nil
					   :fuse nil))))

(defun !gpt2-load-pe (ctx-out ctx wte wpe)
  (call
   (GPT2PositionalEmbedding
    (read-config :n-vocab)
    (read-config :n-ctx)
    (read-config :n-emb))
   ctx wte wpe ctx-out))
	

;; [TODO] Move this into :cl-waffe2/nn package
(defun gpt2-layernorm (x)

  )

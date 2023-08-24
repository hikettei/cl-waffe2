
(in-package :gpt-2-example)

;; Utils

(defun load-npy (path &rest args)
  ;; npz -> AbstractTensor
  (format t "[INFO] load-npy attempts to load ~a...~%" (apply #'format nil path args))
  (parameter (change-facet (numpy-file-format:load-array (apply #'format nil path args)) :direction 'AbstractTensor)))

(defun incf-tensor-ptr (tensor tensor-ptr &key (offset 0))
  (cl-waffe2/backends.cpu::incf-tensor-ptr tensor tensor-ptr :offset offset))

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
	

;; [TODO] Move this into :cl-waffe2/nn package
(defun gpt2-layernorm (x)

  )

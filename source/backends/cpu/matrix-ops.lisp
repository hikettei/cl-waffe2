
(in-package :cl-waffe2/backends.cpu)

;; argmax/argmin/matmul

(declaim (ftype (function (boolean) (signed-byte 8)) trans->c))
(defun trans->c (transpose-specifier)
  (if transpose-specifier
      #.(char-code #\T)
      #.(char-code #\N)))

;; TODO: 2D Gemm
;; TODO: 1D Gemm -> Dot Product
;; TODO: Fix it.
;; TODO: Transpose Tensor.
(defun expand-gemm-form (a b out &key trans-a? trans-b?)
  "[M N] @ [N K] -> [M K]"
  (let ((dtype (dtype out)))
    (case dtype
      (:float
       ;; TODO: Check If The Tensor is continuous on memory.
       (call-with-view
	#'(lambda (a-view b-view c-view)
	    ;; Lazy-Eval when size=symbol.
	    (let ((lda (* `,(stride-of a-view 0)
			  `,(size-of   a-view 1)))
		  (ldb (* `,(stride-of b-view 0)
			  `,(size-of   b-view 1)))
		  (ldc (* `,(stride-of c-view 0)
			  `,(size-of   c-view 1))))
	      ;; a-view = [A.views[n-1], A.views[n]]
	      `(blas-sgemm
		,(trans->c trans-a?)
		,(trans->c trans-b?)
		,(size-of a-view 0)
		,(size-of a-view 1)
		,(size-of b-view 1)
		1.0 ;; alpha
		(tensor-ptr ,a :offset ,(offset-of a-view 0)) ;; a
		,lda ;; LDA
		(tensor-ptr ,b :offset ,(offset-of b-view 0)) ;; B
		,ldb ;; LDB
		0.0 ;; beta
		(tensor-ptr ,out :offset ,(offset-of c-view 0))
		,ldc)))
	`(,a ,b ,out)
	:at-least-dim 2))
      (:double
       (call-with-view
	#'(lambda (a-view b-view c-view)
	    ;; Lazy-Eval when size=symbol.
	    (let ((lda (* `,(stride-of a-view 0)
			  `,(size-of   a-view 1)))
		  (ldb (* `,(stride-of b-view 0)
			  `,(size-of   b-view 1)))
		  (ldc (* `,(stride-of c-view 0)
			  `,(size-of   c-view 1))))
	      ;; a-view = [A.views[n-1], A.views[n]]
	      `(blas-dgemm
		,(trans->c trans-a?)
		,(trans->c trans-b?)
		,(size-of a-view 0)
		,(size-of a-view 1)
		,(size-of b-view 1)
		1.0d0 ;; alpha
		(tensor-ptr ,a :offset ,(offset-of a-view 0)) ;; a
		,lda ;; LDA
		(tensor-ptr ,b :offset ,(offset-of b-view 0)) ;; B
		,ldb ;; LDB
		0.0d0 ;; beta
		(tensor-ptr ,out :offset ,(offset-of c-view 0))
		,ldc)))
	`(,a ,b ,out)
	:at-least-dim 2))
      (T
       (error "The dtype ~a isn't supported yet (TODO)." dtype)))))

(define-impl (MatMulNode :device CPUTensor)
	     :save-for-backward (t t nil)
	     :forward
	     ((self a b out)
	      (let ((trans-a (trans-a? self))
		    (trans-b (trans-b? self)))
		`(,@(expand-gemm-form a b out :trans-a? trans-a :trans-b? trans-b)
		  ,out))))



(in-package :cl-waffe2/backends.cpu)

;; argmax/argmin/matmul

(declaim (ftype (function (boolean) (signed-byte 8)) trans->c))
(defun trans->c (transpose-specifier)
  (if transpose-specifier
      #.(char-code #\T)
      #.(char-code #\N)))

;; Fix: Batched matmul
;; Fix: Matmul with parameter? !t with save-for-backward is working???
;;

;; TODO: 1D Gemm -> Dot Product
;; TODO: Fix it.
;; TODO: Transpose Tensor.
;; FixME: view isn't working at 1d/2d
;; FixME: support row-major with gemm
(defun expand-gemm-form (a b out
			 &key
			   trans-a? trans-b?
			 &aux
			   (a (read-untransposed a))
			   (b (read-untransposed b)))
  "[M N] @ [N K] -> [M K]"
  (let ((dtype (dtype out)))
    (assert (eql (order a) :column)
	    nil
	    "Assertion Failed with (order a) = :column (TODO: Support)")
    (case dtype
      (:float
       ;; TODO: Check If The Tensor is continuous on memory.
       (call-with-view
	#'(lambda (a-view b-view c-view)
	    ;; Lazy-Eval when size=symbol.
	    (let* ((a-view1 (if trans-a?
				(reverse a-view)
				a-view))
		   (m   (size-of c-view 0))
		   (n   (size-of c-view 1))
		   (k   (size-of a-view1 1))
		   (lda (size-of a-view 1))
		   (ldb (size-of b-view 1))
		   (ldc (size-of c-view 1)))
	      ;; [10 12] @ [12 13]
	      ;; [M K] @ [K N]
	      ;; a-view = [A.views[n-1], A.views[n]]
	      `(blas-sgemm
		,(trans->c trans-b?)
		,(trans->c trans-a?)
		,n
		,m
		,k
		1.0 ;; alpha
		(tensor-ptr ,b :offset ,(offset-of b-view 0)) ;; a
		,ldb
		(tensor-ptr ,a :offset ,(offset-of a-view 0)) ;; b
		,lda
		0.0 ;; beta
		(tensor-ptr ,out :offset ,(offset-of c-view 0))
		,ldc)))
	`(,a ,b ,out)
	:at-least-dim 2))
      (:double
       (call-with-view
	#'(lambda (a-view b-view c-view)
	    ;; Lazy-Eval when size=symbol.
	    (let* ((a-view1 (if trans-a?
				(reverse a-view)
				a-view))
		   (m   (size-of c-view 0))
		   (n   (size-of c-view 1))
		   (k   (size-of a-view1 1))
		   (lda (size-of a-view 1))
		   (ldb (size-of b-view 1))
		   (ldc (size-of c-view 1)))
	      ;; [10 12] @ [12 13]
	      ;; [M K] @ [K N]
	      ;; a-view = [A.views[n-1], A.views[n]]
	      `(blas-dgemm
		,(trans->c trans-b?)
		,(trans->c trans-a?)
		,n
		,m
		,k
		1.0d0 ;; alpha
		(tensor-ptr ,b :offset ,(offset-of b-view 0)) ;; a
		,ldb
		(tensor-ptr ,a :offset ,(offset-of a-view 0)) ;; b
		,lda
		0.0d0 ;; beta
		(tensor-ptr ,out :offset ,(offset-of c-view 0))
		,ldc)))
	`(,a ,b ,out)
	:at-least-dim 2))
      (T
       (error "The dtype ~a isn't supported yet (TODO)." dtype)))))

(define-impl (MatMulNode :device CPUTensor
	                 :cache-when-compiled nil
			 :reject-p (supported-dtypes-are 0 :float :double))
	     :save-for-backward (t t nil)
	     :forward
	     ((self a b out)
	      (let ((trans-a (trans-a? self))
		    (trans-b (trans-b? self)))
		`(,@(expand-gemm-form a b out :trans-a? trans-a :trans-b? trans-b)
		  ,out))))


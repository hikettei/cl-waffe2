
(in-package :cl-waffe2/backends.cpu)

;; argmax/argmin/matmul

(declaim (ftype (function (boolean) (signed-byte 8)) trans->c))
(defun trans->c (transpose-specifier)
  (if transpose-specifier
      #.(char-code #\C)
      #.(char-code #\N)))

;; Fix: Batched matmul
;; Fix: Matmul with parameter? !t with save-for-backward is working???
;;

;; TODO: 1D Gemm -> Dot Product
;; TODO: Fix it.
;; TODO: Transpose Tensor.
;; FixME: view isn't working at 1d/2d
;; FixME: support row-major with gemm

(defun expand-gemm-form (a1 b1 c
			 &key
			   trans-a?
			   trans-b?)
  "[M N] @ [N K] -> [M K]"
  (let ((dtype (dtype c))
	(a (if trans-a?
	       (read-untransposed a1)
	       a1))
	(b (if trans-b?
	       (read-untransposed b1)
	       b1)))

    (when trans-a?
      (assert (equal (reverse (last (shape a) 2))
		     (last (shape a1) 2))
	      nil
	      "expand-gemm-form: Assertion Failed."))

    (when trans-b?
      (assert (equal (reverse (last (shape b) 2))
		     (last (shape b1) 2))
	      nil
	      "expand-gemm-form: Assertion Failed."))

    ;; a, b ... untranspsoed tensor
    ;; they're just used to compute strides

    ;; a1, b1 ... tensor with vec, declared in arguments
    
    (assert (eql (order a) :column)
	    nil
	    "Assertion Failed with (order a) = :column (TODO: Support)")

    (call-with-view
     #'(lambda (a-view b-view c-view)
	 (let* ((m (size-of c-view 0))
		(n (size-of c-view 1))
		(k (second (last (shape a1) 2)))
		(k (if (symbolp k)
		       `(cl-waffe2/vm.generic-tensor::read-symbol ',k)
		       k))
		(lda (size-of a-view 1))
		(ldb (size-of b-view 1))
		(ldc (size-of c-view 1)))
	   (case dtype
	     (:float
	      `(blas-sgemm
		,(trans->c trans-b?)
		,(trans->c trans-a?)
		,n
		,m
		,k
		1.0
		
		;; If compile-when-cache = T,
		;; variables that didn't appear in arguments
		;; Is ignored, so (read-untransposed b) is needed to be lazily evaluated.
		
		(tensor-ptr ,b :offset ,(offset-of b-view 0)) ;; no matter which dim=0, dim=1, offsets are common.
		,ldb
		(tensor-ptr ,a :offset ,(offset-of a-view 0))
		,lda
		0.0
		(tensor-ptr
		 ,c
		 :offset ,(offset-of c-view 0))
		,ldc))
	     (:double
	      `(blas-dgemm
		,(trans->c trans-b?)
		,(trans->c trans-a?)
		,n
		,m
		,k
		1.0d0
		;; If compile-when-cache = T,
		;; variables that didn't appear in arguments
		;; Is ignored, so (read-untransposed b) is needed to be lazily evaluated.
		
		(tensor-ptr
		 ,b
		 :offset ,(offset-of b-view 0)) ;; no matter which dim=0, dim=1, offsets are common.
		,ldb
		(tensor-ptr
		 ,a
		 :offset ,(offset-of a-view 0))
		,lda
		0.0d0
		(tensor-ptr
		 ,c
		 :offset ,(offset-of c-view 0))
		,ldc))
	     (T
	      (error "cl-waffe2/backends.cpu: Matmul with OpenBLAS is dedicated to :float or :double, ~a isn't available.
Please consider using another backends." dtype)))))
     `(,a ,b ,c)
     :at-least-dim 2)))

(define-impl (MatMulNode :device CPUTensor
	                 :cache-when-compiled nil ;; TODO: Make it T.
			 :reject-p (supported-dtypes-are 0 :float :double))
	     :save-for-backward (t t nil)
	     :forward
	     ((self a b out)
	      (let ((trans-a (trans-a? self))
		    (trans-b (trans-b? self)))
		`(,@(expand-gemm-form a b out :trans-a? trans-a :trans-b? trans-b)
		  ,out))))



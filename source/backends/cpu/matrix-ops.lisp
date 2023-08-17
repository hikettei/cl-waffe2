
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
			 a-ptr b-ptr o-ptr
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
	      "expand-gemm-form: Assertion Failed 1: ~a ~a."
	      (shape a) (shape a1)))

    (when trans-b?
      (assert (equal (reverse (last (shape b) 2))
		     (last (shape b1) 2))
	      nil
	      "expand-gemm-form: Assertion Failed 2: ~a ~a."
	      (shape b) (shape b1)))

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
		
	        (incf-tensor-ptr ,b1 ,b-ptr :offset ,(offset-of b-view 0)) ;; no matter which dim=0, dim=1, offsets are common.
		,ldb
		(incf-tensor-ptr ,a1 ,a-ptr :offset ,(offset-of a-view 0))
		,lda
		0.0
		(incf-tensor-ptr
		 ,c
		 ,o-ptr
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
		
		(incf-tensor-ptr
		 ,b1
		 ,b-ptr
		 :offset ,(offset-of b-view 0)) ;; no matter which dim=0, dim=1, offsets are common.
		,ldb
		(incf-tensor-ptr
		 ,a1
		 ,a-ptr
		 :offset ,(offset-of a-view 0))
		,lda
		0.0d0
		(incf-tensor-ptr
		 ,c
		 ,o-ptr
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
		    (trans-b (trans-b? self))
		    (a-ptr   (gensym "PTR"))
		    (b-ptr   (gensym "B"))
		    (o-ptr   (gensym "OUT")))
		`(locally (declare (optimize (speed 1)))
		   (with-tensor-ptrs ((,a-ptr ,a)
				      (,b-ptr ,b)
				      (,o-ptr ,out))
		     ,(expand-gemm-form a b out a-ptr b-ptr o-ptr :trans-a? trans-a :trans-b? trans-b)
		     ;; Sometime matmul fails due to wrong arguments
		     ;; But proceeds with no errors...
		     ,out)))))

;; [TODO] Optimize to get more speed!
(defun expand-arg-maxmin-form (x out type x-ptr &aux (index (gensym)))
  (declare (type (and keyword (member :max :min)) type))
  `(let (;;(x-vec (tensor-vec ,x))
	 (o-vec (tensor-vec ,out))
	 (,index 0))
     (declare (type (unsigned-byte 32) ,index)
	     ;; (type (simple-array ,(dtype->lisp-type (dtype x)) (*)) x-vec)
	      (type (simple-array ,(dtype->lisp-type (dtype out)) (*)) o-vec))
	      
     ,(call-with-view
       #'(lambda (x-view o-view)
	   `(progn
	      (setf (aref o-vec ,index)
		    ;; (blas-iXmax size ptr stride) -> max/min value
		    ;; indices start from 1, if 0, there's any errors
		    (1-
		     (the fixnum
			  (,(case (dtype x)
			      (:double
			       (case type
				 (:max 'blas-idmax)
				 (:min 'blas-idmin)))
			      (:float
			       (case type
				 (:max 'blas-ismax)
				 (:min 'blas-ismin)))
			      (t (error "CPUTensor: argmax/argmin/max/min do not support ~a. only :float and :double" (dtype x))))
			   ,(size-of x-view 0)
			   (incf-tensor-ptr ,x ,x-ptr :offset ,(offset-of x-view 0))
			   ,(stride-of x-view 0)))))
	      (incf ,index ,(stride-of o-view 0))))
       `(,x ,out)
       :at-least-dim 1
       :force-order t)))

;; TODO: ArgMax for SparseMatrix
(define-impl (ArgMax-Node :device CPUTensor)
	     :forward ((self x out)
		       (let ((x-ptr (gensym "PTR")))
			 `(with-tensor-ptr (,x-ptr ,x)
			    (locally (declare (optimize (speed 1)))
			      ,(expand-arg-maxmin-form x out :max x-ptr)
			      ,out)))))

(define-impl (ArgMin-Node :device CPUTensor)
	     :forward ((self x out)
		       (let ((x-ptr (gensym "PTR")))
			 `(with-tensor-ptr (,x-ptr ,x)
			    (locally (declare (optimize (speed 1)))
			      ,(expand-arg-maxmin-form x out :min x-ptr)
			      ,out)))))

;; [TODO] Force it using SIMD, by writing C kernel directly.
(defun expand-maxmin-value-form (x out type x-ptr &aux (index (gensym)))
  (declare (type (and keyword (member :max :min)) type))
  `(let ((x-vec (tensor-vec ,x))
	 (o-vec (tensor-vec ,out))
	 (,index 0))
     (declare (type (unsigned-byte 32) ,index)
	      (type (simple-array ,(dtype->lisp-type (dtype x)) (*)) x-vec)
	      (type (simple-array ,(dtype->lisp-type (dtype out)) (*)) o-vec))

     (macrolet ((%* (x y) `(the fixnum (* (the fixnum ,x) (the fixnum ,y)))))
       ,(call-with-view
	 #'(lambda (x-view o-view)
	     `(progn
		(setf
		 (aref o-vec ,index)
		 (aref x-vec
		       (+
			,(offset-of x-view 0)
			(%* ,(stride-of x-view 0)
			    (1-
			     (the fixnum
				  (,(case (dtype x)
				      (:double
				       (case type
					 (:max 'blas-idmax)
					 (:min 'blas-idmin)))
				      (:float
				       (case type
					 (:max 'blas-ismax)
					 (:min 'blas-ismin)))
				      (t (error "CPUTensor: argmax/argmin/max/min do not support ~a. only :float and :double" (dtype x))))
				    ,(size-of x-view 0)
				    (incf-tensor-ptr ,x ,x-ptr :offset ,(offset-of x-view 0))
				    ,(stride-of x-view 0))))))))
		(incf ,index ,(stride-of o-view 0))))
	 `(,x ,out)
	 :at-least-dim 1
	 :force-order t))))

(defun expand-simd-maxmin-value-form (x out type x-ptr out-ptr)
  (declare (type (and string (member "max" "min")))
	   (type AbstractTensor x out)
	   (type symbol x-ptr out-ptr))
  (let ((fname (make-fname (dtype x) type)))
    (call-with-view
     #'(lambda (x-view o-view)
	 `(,fname ,(size-of x-view 0) (incf-tensor-ptr ,x ,x-ptr :offset ,(offset-of x-view 0)) ,(stride-of x-view 0) (incf-tensor-ptr ,out ,out-ptr :offset ,(offset-of o-view 0)) ,(stride-of o-view 0)))
     `(,x ,out)
     :force-order t
     :at-least-dim 1)))

;; [TODO] reject for sparse matrices
(define-impl (MaxValue-Node :device CPUTensor
	      :reject-p #'simd-extension-p)
	     :forward ((self x out)
		       (let ((x-ptr (gensym "PTR"))
			     (out-ptr (gensym "PTR")))
			 `(with-tensor-ptrs ((,x-ptr ,x)
					     (,out-ptr ,out))
			    (locally (declare (optimize (speed 1)))
			      ,(if *simd-extension-p*
				   (expand-simd-maxmin-value-form x out "max" x-ptr out-ptr)
				   (expand-maxmin-value-form x out :max x-ptr))
			      ,out)))))

(define-impl (MinValue-Node :device CPUTensor
	      :reject-p #'simd-extension-p)
	     :forward ((self x out)
		       (let ((x-ptr (gensym "PTR"))
			     (out-ptr (gensym "PTR")))
			 `(with-tensor-ptrs ((,x-ptr ,x)
					     (,out-ptr ,out))
			    (locally (declare (optimize (speed 1)))
			      ,(if *simd-extension-p*
				   (expand-simd-maxmin-value-form x out "min" x-ptr out-ptr)
				   (expand-maxmin-value-form x out :min x-ptr))
			      ,out)))))



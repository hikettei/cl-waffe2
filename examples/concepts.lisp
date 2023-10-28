
(defpackage :concepts
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/nn
   :cl-waffe2/distributions
   :cl-waffe2/base-impl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/vm
   :cl-waffe2/optimizers

   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.lisp
   ))

(in-package :concepts)

;; Section0. Why not: Keep Using Python?
(defsequence NeuralNetwork ()
	     "My Neural Network"
	     (asnode #'!flatten)
	     (LinearLayer (* 28 28) 512)
	     (asnode #'!relu)
	     (LinearLayer 512 512)
	     (asnode #'!relu)
	     (LinearLayer 512 10))


;; Section1. Nodes are Abstract, Lazy, and Small.
(defnode (MatMulNode-Revisit (self)
	  :where (A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :backward ((self dout a b c)
		     (declare (ignore c))
		     (values
		      (!matmul dout (!t b))
		      (!matmul (!t a) dout)
		      nil))
	  :documentation "OUT <- GEMM(1.0, A, B, 0.0, C)"))

(defclass MyTensor (CPUTensor) nil)

;; The performance would be the worst. Should not be used for practical.
(defun gemm! (m n k a-offset a b-offset b c-offset c)
  "Computes 1.0 * A[M N] @ B[N K] + 0.0 * C[M K] -> C[M K]"
  (declare (type (simple-array single-float (*)) a b c)
	   (type (unsigned-byte 32) m n k a-offset b-offset c-offset)
	   (optimize (speed 3) (safety 0)))
  (dotimes (mi m)
    (dotimes (ni n)
      (dotimes (ki k)
	(setf (aref c (+ c-offset (* mi K) ni))
	      (* (aref a (+ a-offset (* mi n) ki))
		 (aref b (+ b-offset (* ki k) ni))))))))

(define-impl (MatmulNode-Revisit :device MyTensor)
	     :save-for-backward (t t nil)
	     :forward ((self a b c)
		       `(,@(call-with-view
			    #'(lambda (a-view b-view c-view)
				`(gemm!
				  ,(size-of a-view 0)
				  ,(size-of b-view 0)
				  ,(size-of c-view 1)
				  ,(offset-of a-view 0) (tensor-vec ,a)
				  ,(offset-of b-view 0) (tensor-vec ,b)
				  ,(offset-of c-view 0) (tensor-vec ,c)))
			    `(,a ,b ,c)
			    :at-least-dim 2)
			 ,c)))

;; Gemm with Lisp
(defun test-gemm (&key (bench nil))
  (with-devices (MyTensor)
    (let ((a (randn `(100 100)))
	  (b (randn `(100 100)))
	  (c (make-input `(100 100) nil)))
      
      (proceed
       (call (MatmulNode-Revisit) a b c)
       :measure-time bench))))


;; Gemm with OpenBLAS
;; Set bench=t, and measures time
;; As expected, this one is 20~30times faster.
(defun test-gemm-cpu (&key (bench nil))
  (with-devices (CPUTensor)
    (let ((a (randn `(100 100)))
	  (b (randn `(100 100)))
	  (c (make-input `(100 100) nil)))

      (proceed
       (call (MatmulNode :float) a b c)
       :measure-time bench))))

(proceed-time (!add 1 1))

(proceed-bench (!softmax (randn `(100 100))))

;; Let cl-waffe2 recognise MyTensor is a default device
;; And the priority is: MyTensor -> CPUTensor -> LispTensor
(set-devices-toplevel 'MyTensor 'CPUTensor 'LispTensor)

(let ((a (make-input `(A B) :A))
      (b (make-input `(A B) :B)))
  (let ((model (build (!sum (!mul a b)) :inputs `(:A :B))))
    (print model)
    ;; model is a compiled function: f(a b)
    (forward model (randn `(3 3)) (randn `(3 3)))))

(defun my-matmul (a b)
  (let* ((m (first  (shape a)))
	 (k (second  (shape b)))
	 
	 (c (make-input `(,m ,k) nil))) ;; <- InputTensor
    (call (MatmulNode-Revisit) a b c)))

(print
 (proceed (my-matmul (randn `(3 3)) (randn `(3 3)))))

(node->defun %mm-softmax (A[m n] B[n k] -> C[m k])
  (!softmax (my-matmul a b)))

;; JIT Enabled Matrix Operations
(defun local-cached-matmul ()
  ;; Works like Lisp Function
  (print (time (%mm-softmax (randn `(3 3)) (randn `(3 3)))))
  (print (time (%mm-softmax (randn `(3 3)) (randn `(3 3))))))

(defun build-usage ()
  (let ((a (make-input `(A B) :X))
	(b (make-input `(A B) :Y)))

    (let ((compiled-model (build (!sum (!mul a b)) :inputs `(:X :Y))))
      (print compiled-model)
      (forward compiled-model (randn `(3 3)) (randn `(3 3))))))

;; Section2. Advanced Network Configurations

;;; Composite
(defmodel (LayerNorm-Revisit (self normalized-shape &key (eps 1.0e-5) (affine T))
	   :slots ((alpha :initform nil :accessor alpha-of)
		   (beta  :initform nil :accessor beta-of)
		   (shape :initform nil :initarg :normalized-shape :accessor dim-of)
		   (eps   :initform nil :initarg :eps :accessor eps-of))
	   ;; Optional
	   :where (X[~ normalized-shape] -> out[~ normalized-shape])
	   :on-call-> layer-norm)

  ;; Constructor
  (when affine
    (setf (alpha-of self) (parameter (ax+b `(,@normalized-shape) 0 1))
	  (beta-of  self) (parameter (ax+b `(,@normalized-shape) 0 0)))))

(defmethod layer-norm ((self LayerNorm-Revisit) x)
  (with-slots ((alpha alpha) (beta beta)) self
    (let* ((last-dim (length (dim-of self)))
	   (u (!mean x :axis (- last-dim) :keepdims t))
	   (s (!mean (!expt (!sub x u) 2) :axis (- last-dim) :keepdims t))
	   (x (!div (!sub x u)
		    (!sqrt (!add (->contiguous s) (eps-of self))))))

      (if (and alpha beta)
	  ;; both inserts broadcastable axis
	  (!add (!mul x (%transform alpha[i] -> [~ i]))
	        (!flexible beta)) ;; !flexible = (%transform alpha[i] -> [~ i])
	  x))))

(print (call (LayerNorm-Revisit `(10)) (randn `(10 10 10))))

;; One more example
(defmodel (Softmax-Model (self)
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (let* ((x1 (!sub x (!mean x  :axis 1 :keepdims t)))
                              (z  (!sum   (!exp x1) :axis 1 :keepdims t)))
                         (!div (!exp x1) z)))))

(defmodel-as (Softmax-Model)
  :where (A[~] -> B[~])
  :asif :function :named %softmax)

;; Section3 Make everything user-extensible

;;; Customized Autodiff

;;; The simplest case of ScalarTensor

(defclass MyScalarTensor (ScalarTensor) nil)
(set-devices-toplevel 'MyTensor 'CPUTensor 'LispTensor 'MyScalarTensor)


(define-op (MyMul (self)
	    :where (A[scal] B[scal] -> A[scal] where scal = 1)
	    :out-scalar-p t
	    :save-for-backward-names (a b)
	    :forward ((self a b)
		      (with-setting-save4bw ((a a) (b b))
			(setf (tensor-vec a) (* (tensor-vec a) (tensor-vec b)))
			a))
	    :backward ((self dy)
		       (with-reading-save4bw ((a a) (b b))
			 (values
			  (make-tensor
			   (* (tensor-vec dy) (tensor-vec b)))
			  (make-tensor
			   (* (tensor-vec dy) (tensor-vec a))))))))

(define-op (MySin (self)
	    :where (A[scal] out[scal] -> out[scal] where scal = 1)
	    :out-scalar-p t
	    :save-for-backward-names (a)
	    :forward ((self a out)
		      (with-setting-save4bw ((a a))
			(setf (tensor-vec out) (sin (tensor-vec a)))
			out))
	    :backward ((self dy)
		       (with-reading-save4bw ((a a))
			 (values
			  (make-tensor
			   (* (tensor-vec dy)
			      (cos (tensor-vec a))))
			  nil)))))
(defun !mymul (a b)
  (call (MyMul) a b))

(defun !mysin (x)
  (call (MySin) x (make-clone x)))

(defun try-original-autodiff ()
  (let ((a (parameter (make-tensor 1))))
    (proceed-backward (!mysin (!mysin a)))
    (grad a)))

;;; Differentiable Programming

(defoptimizer (MySGD (self param &key (lr 1e-3))
	       :slots ((lr :initarg :lr :reader sgd-lr))))

(node->defun %step-sgd (Param[~] Grad[~] Lr[scal] -> Param[~] where scal = 1)
  (A-=B param (A*=scal grad lr)))

(defmethod step-optimize ((optimizer MySGD))
  (let* ((lr    (make-tensor (sgd-lr optimizer)))
	 (param (read-parameter optimizer))
	 (grad  (grad param)))
    (with-no-grad
      (%step-sgd param grad lr))))

(defun simple-opt-model ()
  (let* ((loss (!mean (!matmul (parameter (randn `(3 3)))
			       (parameter (randn `(3 3))))))
	 (model (build loss)))

    (mapc (hooker x (MySGD x :lr 1e-3)) (model-parameters model))
    (forward model)
    (backward model)
    (mapc #'call-optimizer! (model-parameters model))))
#|
({MYTENSOR[float] :shape (3 3) -> :view (<T> <T>) -> :visible-shape (3 3)  
  ((0.25052267  -0.16212857 -1.3183842)
   (-1.078968   0.27860558  0.40701634)
   (-0.10987697 -1.2562615  0.6179133))
  :facet :exist
  :requires-grad T
  :optimizer <AbstractOptimizer: MYSGD() -> TID12604>}
 {MYTENSOR[float] :shape (3 3) -> :view (<T> <T>) -> :visible-shape (3 3)  
  ((-0.5223165  2.3579814   0.13172081)
   (0.57671905  0.56324756  1.1230979)
   (0.10274803  0.008530198 1.7588508))
  :facet :exist
  :requires-grad T
  :optimizer <AbstractOptimizer: MYSGD() -> TID12610>})
|#

;; Section4 Loop Optimization By Metaprogramming

;; Section5 Graph-Level Optimization

;; Section6 Interop: Common Lisp Array and AbstractTensor

;; Section7 Visualize, eazy to debug

;; Section8 JIT/Caching

;; Section9 Symbolic Differentiation and Device Specific Optimization

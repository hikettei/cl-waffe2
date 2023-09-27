;;;; © 2023-2023 hikettei

(in-package :cl-user)

(load "cl-waffe2.asd")
(ql:quickload :cl-waffe2)

(defpackage :examples
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
   :cl-waffe2/backends.lisp))

(in-package :examples)

;; For List Meetup
;; TO ADD: Symbolic Diff

;; 1. Basic Developling Cycle
;;  - `Call` creates computation nodes lazily.
;;        -> The `cl-waffe2/base-impl` package provides convenient wrappers (e.g.: !add !+ !matmul ...)
;;
;;  -  Observe the result with `Proceed` function.
;;

(call (AddNode :float) (randn `(3 3)) (randn `(3 3)))

(!add (randn `(3 3)) (randn `(3 3)))

(proceed (!add (randn `(3 3)) (randn `(3 3))))

;; 2. AbstractTensor

(show-backends)

(with-devices (CPUTensor LispTensor)
  (randn `(3 3)))

(with-devices (LispTensor CPUTensor)
  (!matmul (randn `(3 3)) (randn `(3 3))))

(defclass MyTensor (LispTensor) nil)

(with-devices (MyTensor)
  (beta `(10 10) 0.5 0.5))

;; 3. Extensible Facet APIs

;; Translating AbstractTensor <-> Any other forms of matrices

(change-facet #2A((1 2 3)
		  (4 5 6)
		  (7 8 9))
	      :direction 'AbstractTensor)

(change-facet (randn `(3 3)) :direction 'array)

(change-facet (randn `(3 3)) :direction 'simple-array)

;; No copies are created; all modifications are synchronized.
(let ((a (randn `(3 3))))
  (with-facet (a* (a :direction 'array))
    ;; Fill diagoals with 0.0
    (setf (aref a* 0 0) 0.0)
    (setf (aref a* 1 1) 0.0)
    (setf (aref a* 2 2) 0.0)
    (print a*))
  (print a))

;; 4. AbstractNode

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

;; Declares AbstractNode
(defnode (MatMulNode-Revisit (self)
	  :where (A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :backward ((self dout a b c)
		     (declare (ignore c))
		     (values
		      (!matmul dout (!t b))
		      (!matmul (!t a) dout)
		      nil))
	  :documentation "OUT <- GEMM(1.0, A, B, 0.0, C)"))

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

(defun !matmul-revisit (a b)
  ;; A[M N] @ B[N K] -> C[M K]
  (let* ((~ (butlast (shape a) 2))
	 (out (make-input `(,@~
			    ,(car    (last (shape a) 2))
			    ,(second (last (shape b) 2)))
			  nil)))
    (call (MatmulNode-Revisit) a b out)))

(with-devices (MyTensor)
  ;; Node creation
  (print (MatmulNode-Revisit))

  ;; ↓ Intentionally produces Shaping-Error
  ;;(print (!matmul-revisit (randn `(10 2)) (randn `(3 10))))

  (print (!matmul-revisit (randn `(10 2)) (randn `(2 10))))

  ;; ~ = inf
  ;; gemm! is now a function which can be applied into ND Array
  (print
   (proceed
    (!matmul-revisit (randn `(10 2 2))
		     (randn `(10 2 2))))))

;; Of course, (deifne-impl (MatmulNode-Revisit :device CPUTensor)) is also ok. cl-waffe2 is intended so.
;; See also: `./source/backends/lisp/logical.lisp` for intuitive examples of define-impl-op combined with a do-compiled-loop macro.


;; 5. define-impl-op

;; define-op = defnode + define-impl-op
;; Creating a graph-processing library which is completely separeted from cl-waffe2 ecosystem

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

(print (try-original-autodiff)) ;; nearly equals (* (cos 1) (cos (sin 1)))

;; Composite defmodel


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

(print (LayerNorm-Revisit `(10)))
(print (call (LayerNorm-Revisit `(10)) (randn `(10 10 10))))

;; Nodes -> Function

;; With appropriate info of Ranks/Devices/Tensors, cl-waffe2 can compile nodes and cache it.
;; Such functions behave as if they're truly functions where do not require being compiled.

(defmodel (Softmax-Model (self)
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (let* ((x1 (!sub x (!mean x  :axis 1 :keepdims t)))
                              (z  (!sum   (!exp x1) :axis 1 :keepdims t)))
                         (!div (!exp x1) z)))))

(defmodel-as (Softmax-Model)
  :where (A[~] -> B[~])
  :asif :function :named %softmax)

(print (time (%softmax (randn `(100 100)))))

(node->defun L1Loss-Revisit (A[~] B[~] -> C[~])
  (->scal (!sum (!- a b))))

(print (time (L1Loss-Revisit (randn `(300 300)) (randn `(300 300)))))

(defun doing-some-stuff (x)
  (funcall (node->lambda (A[~] -> A[i] where i = 1)
	     (->scal (!sum a)))
	   x))

(print (time (doing-some-stuff (randn `(100 100)))))

;; Debugging: Disassemble

(disassemble-waffe2-ir (!softmax
			(parameter (randn `(10 10)))
			:avoid-overflow nil))

;; Profiler
(proceed-bench
 (!softmax (randn `(100 100)))
 :n-sample 100)

;; Diff
(proceed-bench
 (call (Conv2D 3 6 `(5 5)) (randn `(10 3 25 25))))

;; Optional Broadcasting

;; Should pass with libraries which adapts Numpy-Semantic Broadcasting
;; But produces Shaping-Error in cl-waffe2:

(!add (randn `(3 3)) (randn `(3)))

;; Explicting Broadcastable Axis

;; In order to automatically use broadcasting rank-up rule at arbitary axis, both of conditions below must be satisfied:
;; - AbstractNode: Corresponding axis must be ~
;; - Tensor:       Corresponding axis must be <1 x N>

(print (!flexible (randn `(3 3)) :at 0))
(print (!flexible (randn `(3 3)) :at 1))
(print (!flexible (randn `(3 3)) :at 2))


(print
 (proceed
  (!add
   (ax+b `(3 3) 0 0)
   (!flexible (ax+b `(3) 1 0) :at 0))))

(print
 (proceed
  (!add
;   (ax+b `(3 3) 0 0)
   (!flexible (ax+b `(3) 1 0) :at -1))))

;; %transform: Aliases for Reshape/View/Permute/Flexible

;; !Flexible
(let ((a (ax+b `(3 3) 1 0)))
  (print (%transform a[i j] -> [i ~ j])))

(print (%transform (ax+b `(3 3) 1 0)[i j] -> [i ~ j]))

;; Permute
(let ((a (ax+b `(3 3) 1 0)))
  (print (%transform a[i j] -> [j i])))

;; View
(let ((a (ax+b `(1 3) 1 0)))
  (print (%transform A[i j] -> [*3 t])))

;; Slicing
(let ((a (ax+b `(3 5 2) 1 0)))
  (print (%transform a[i j k] -> [(0 m) t 1] where m = 3)))

;; Example: Adding Biases column-wise to the given tensor
(defun affine (a x b)
  "Applies Affine Transformation. A = [m], X = [~ m] B = [m]"
  (!add (!mul
	 (%transform a[i] -> [~ i])
	 x)
	(%transform b[i] -> [~ i])))

;; call-> and asnode function.

(print (asnode #'!sin))
(print (call (asnode #'!sin) (randn `(3 3))))
;; X += 1
(print
 (proceed
  (call
   (asnode #'!add 1.0)
   (ax+b `(3 3) 1 0))))

(defparameter *result* nil)

;; call->
(print
 (call-> (randn `(1 20))
	 (LinearLayer 20 10)
	 (asnode #'!relu)
	 (LinearLayer 10 5)
	 (asnode #'!softmax :avoid-overflow nil)
	 (asnode #'!view 0 0)))

(defsequence MNIST-CNN (&key
			(out-channels1 4)
			(out-channels2 16))
	     (Conv2D 1 out-channels1 `(3 3))
	     (asnode #'!relu)     
	     (MaxPool2D    `(2 2))
	     (Conv2D out-channels1 out-channels2 `(5 5))
	     (asnode #'!relu)
	     (MaxPool2D `(2 2))
	     (asnode #'!reshape t (* 16 4 4)) 
	     (LinearLayer (* 16 4 4) 10))

(print (MNIST-CNN))

;; See also: ./gpt-2/model.lisp

;; Mathematical Optimizing

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

;; Customized Printing
(defmethod on-print-object ((optimizer MySGD) stream)
  (format stream "lr=~a" (sgd-lr optimizer)))

(print (MySGD (parameter (randn `(3 3)))))

;; hook-optimizer!
;; call-optimizer!

(defun simplest-optimizing-model ()
  (let* ((loss (!mean (!matmul (parameter (randn `(3 3)))
			       (parameter (randn `(3 3))))))
	 (model (build loss)))

    (mapc (hooker x (MySGD x :lr 1e-3)) (model-parameters model))
    
    (forward model)
    (backward model)

    (mapc #'call-optimizer! (model-parameters model))))

(simplest-optimizing-model)

;; See also: MNIST?
;; That's all


(node->defun ReLURevisit (A[~] -> B[~])
  (!relu (!t a)))

(print (ReLURevisit (ax+b `(3 3) 1 0)))

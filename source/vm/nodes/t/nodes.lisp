
(in-package :cl-waffe2/vm.nodes.test)

(in-suite :test-nodes)

(defnode (Bijective-Function (myself)
	  :where (A[x y] B[x y] -> A[x y])
	  :documentation "Bijective-Function has a one-to-one correspondence."))

(defnode (Transpose-Function (myself)
	  :where (A[x y] -> B[y x])
	  :documentation "x y -> y x"))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defclass MyBackend (AbstractTensor) nil)
  (defclass MyBackend-With-Impl (AbstractTensor) nil))

(define-impl (Bijective-Function :device CPUTensor)
	     :forward ((self x y)
		       `(values ,x ,y))
	     :backward ((self dout dx dy)
			(declare (ignore dx dy))
			(values dout dout)))

(define-impl (Transpose-Function :device CPUTensor)
	     :forward ((self x)
		       `(progn ,x))
	     :backward ((self dout dx)
			(declare (ignore dx))
			(values dout)))


(define-impl (Bijective-Function :device MyBackend-With-Impl)
	     :forward ((self x)
		       `(progn ,x))
	     :backward ((self dout dy)
			(declare (ignore dy))
			(values dout)))

(defun test-switch-backend1 ()
  (with-devices (CPUTensor)
    (typep (Bijective-Function) 'CL-WAFFE2/VM.NODES.FACETS-TMP::BIJECTIVE-FUNCTION-CPUTENSOR)))

(defun test-switch-backend2 ()
  (with-devices (MyBackend CPUTensor)
    (typep (Bijective-Function) 'CL-WAFFE2/VM.NODES.FACETS-TMP::BIJECTIVE-FUNCTION-CPUTENSOR)))

(defun test-switch-backend3 ()
  (with-devices (MyBackend-with-impl CPUTensor)
    (typep (Bijective-Function) 'CL-WAFFE2/VM.NODES.FACETS-TMP::BIJECTIVE-FUNCTION-MYBACKEND-WITH-IMPL)))


(test heuristic-backend-dispatching-test
  (is (test-switch-backend1))
  (is (test-switch-backend2))
  (is (test-switch-backend3)))

(defun shape-test ()
  (let ((out (forward (Transpose-Function)
		      (forward (Bijective-Function)
			       (make-tensor `(10 3))
			       (make-tensor `(10 3))))))
    (equal (shape out) `(3 10))))

(test shape-test
  (is (shape-test)))


;; [Add] defnode, define-impl, define-impl-op
;; [Add] define-op
;; [Add] Nodes which return multiple arguments, are working well? -> Later, do it in the composite too

(defnode (OpAddTest-Scal (self)
	  :out-scalar-p t
	  :where (A[scal] B[scal] -> A[scal] where scal = 1)))

(define-impl-op (OpAddTest-Scal)
		:forward ((self x y)
			  (make-tensor
			   (+ (tensor-vec x)
			      (tensor-vec y))))
		:backward ((self dy x y)
			   (declare (ignore x y))
			   (values dy dy)))

(test define-impl-op-forward-test
  (is (= 2 (tensor-vec
	    (proceed
	     (forward
	      (OpAddTest-Scal)
	      (make-tensor 1)
	      (make-tensor 1)))))))

(define-op (OpMulTest-Scalar (self)
	    :out-scalar-p t
	    :where (A[scal] B[scal] -> A[scal] where scal = 1)
	    :save-for-backward-names (a b)
	    :forward ((self a b)
		      (set-save-for-backward self 'a a)
		      (set-save-for-backward self 'b b)		      
		      (make-tensor
		       (* (tensor-vec a)
			  (tensor-vec b))))
	    :backward ((self dy)
		       (with-reading-save4bw ((a a)
					      (b b))
			 (values
			  (make-tensor
			   (* (tensor-vec b)
			      (tensor-vec dy)))
			  (make-tensor
			   (* (tensor-vec a)
			      (tensor-vec dy))))))))

(defun diff-test-proceed ()
  (let ((a (parameter (make-tensor 2)))
	(b (parameter (make-tensor 3))))

    (proceed-backward
     (call (OpMulTest-Scalar) a b))
    
    (and
     (= (tensor-vec (grad a)) 3)
     (= (tensor-vec (grad b)) 2))))


;; This is NOT WORKING...
(defun diff-test-vm ()
  (let ((a (parameter (make-tensor 2)))
	(b (parameter (make-tensor 3))))

    (let ((model (build (call (OpMulTest-Scalar) a b))))
      (forward model)
      (backward model))
    (and
     (= (tensor-vec (grad a)) 3)
     (= (tensor-vec (grad b)) 2))))
     
(test define-op-diff-test
  (is (= 6
	 (tensor-vec
	  (proceed
	   (call (OpMulTest-Scalar) (make-tensor 2) (make-tensor 3))))))
  (is (let ((a (parameter (make-tensor 2)))
	    (b (parameter (make-tensor 3))))
	(proceed-backward
	 (call (OpMulTest-Scalar) a b))
	(and
	 (= 3 (tensor-vec (grad a)))
	 (= 2 (tensor-vec (grad b))))))
  (is (diff-test-proceed))
  (is (diff-test-vm)))

(define-op (SwapNode (self)
	    :out-scalar-p t
	    :where (A[scal] B[scal] -> B[scal] A[scal] where scal = 1)
	    :save-for-backward-names (a b)
	    :forward ((self a b)
		      (with-setting-save4bw ((a a) (b b))
			(values b a)))
	    :backward ((self dy)
		       (with-reading-save4bw ((a a) (b b))
			 (values b a)))))

(defun multiple-value-return-test ()
  (multiple-value-bind (b a)
      (call (SwapNode) (make-tensor 4) (make-tensor 2))
    (let ((out (proceed (!add a b))))
      (= (tensor-vec out) 6))))

(defun multiple-value-return-test-vm ()
  (multiple-value-bind (b a)
      (call (SwapNode) (make-tensor 4) (make-tensor 2))
    (let ((out (forward (build (!add a b)))))
      (= (tensor-vec out) 6))))





(in-package :cl-waffe2/base-impl.test)

;; Testing APIs provides by cl-waffe2/base-impl

;; !add !sub !mul !div
;; reshape proceed proceeed-backward !view ->scal ->mat
;; permute with proceeded result.

(defmacro lproceed (tensor)
  `(progn;;with-devices (cl-waffe2/backends.lisp:LispTensor)
     (proceed ,tensor)))

;; =============================================================
;; Testing general-purpose arithmetic APIs: !add !sub !mul !div.
;; =============================================================

(deftest test-arithmetic-form
  (testing "Scalar and Scalar"
    ;; Scalar And Scalar
    (ok (= (tensor-vec
	    (proceed (!add 1.0 1.0)))
	   2.0))
    (ok (= (tensor-vec
	    (proceed (!sub 1.0 1.0)))
	   0.0))
    (ok (= (tensor-vec
	    (proceed (!mul 2.0 3.0)))
	   6.0))
    (ok (= (tensor-vec
	    (proceed (!div 6.0 3.0)))
	   2.0)))


  (testing "Scalar and Matrix"
    (ok (= (vref
	    (proceed (!add (make-tensor `(10 10) :initial-element 1.0) 1.0))
	    0)
	   2.0))
    (ok (= (vref
	    (proceed (!sub (make-tensor `(10 10) :initial-element 1.0) 1.0))
	    0)
	   0.0))
    (ok (= (vref
	    (proceed (!mul (make-tensor `(10 10) :initial-element 2.0) 3.0))
	    0)
	   6.0))
    (ok (= (vref
	    (proceed (!div (make-tensor `(10 10) :initial-element 6.0) 3.0))
	    0)
	   2.0)))
  ;; Matrix and Scalar
  (testing "Matrix and Scalar"
    (ok (= (vref
	    (proceed (!add 1.0 (make-tensor `(10 10) :initial-element 1.0)))
	    0)
	   2.0))    
    (ok (= (vref
	    (proceed (!sub 1.0 (make-tensor `(10 10) :initial-element 1.0)))
	    0)
	   0.0))    
    (ok (= (vref
	    (proceed (!mul 2.0 (make-tensor `(10 10) :initial-element 3.0)))
	    0)
	   6.0))    
    (ok (= (vref
	    (proceed (!div 6.0 (make-tensor `(10 10) :initial-element 3.0)))
	    0)
	   2.0)))

  ;; Matrix and matrix
  (testing "Matrix and Matrix"
    (ok (= (vref
	    (proceed (!add (make-tensor `(10 10) :initial-element 1.0) (make-tensor `(10 10) :initial-element 1.0)))
	    0)
	   2.0))

    (ok (= (vref
	    (proceed (!sub (make-tensor `(10 10) :initial-element 1.0) (make-tensor `(10 10) :initial-element 1.0)))
	    0)
	   0.0))

    (ok (= (vref
	    (proceed (!mul (make-tensor `(10 10) :initial-element 2.0) (make-tensor `(10 10) :initial-element 3.0)))
	    0)
	   6.0))

    (ok (= (vref
	    (proceed (!div (make-tensor `(10 10) :initial-element 6.0) (make-tensor `(10 10) :initial-element 3.0)))
	    0)
	   2.0))))

;; =============================================================
;; Testing !reshape ->scal ->mat !unsqueeze !squeeze
;; =============================================================

;; Note: !view ... testing with !sum !mean

(deftest reshaping-test
  (testing "Reshape [Shape=Fixed]"
    (ok (equal (shape (!reshape (make-tensor `(10 10)) 100))
	       `(100)))
    (ok (equal (shape (!reshape (make-tensor `(10 10)) t))
	       `(100)))))

(deftest ->scal-test
  (testing "->scal"
    (ok (scalar-p (->scal (make-tensor `(1 1)))))))

(deftest ->mat-test
  (testing "->mat"
    (ok (equal `(1)   (shape (->mat (make-tensor 1.0)))))
    (ok (equal `(1 1) (shape (->mat (make-tensor 1.0) :dims 2))))))

;; =============================================================
;; Testing proceed proceed-backward
;; =============================================================

;; =============================================================
;; Testing !flexible, broadcasting
;; =============================================================

;; TODO: It isn't enough
(deftest flexible-test
  (testing "Adding an infinite-rank (~) to 1D tensor."
    (ok (!add (make-tensor `(10 10)) (!flexible (make-tensor `(10)))))))


;; =============================================================
;; Testing Differentiable Proceed
;; =============================================================

;; out = C + A * B
(defun proceed-continue-test (n)
  (let ((a (parameter (ax+b `(10 10) 0 3)))
	(b (parameter (ax+b `(10 10) 0 2)))
	(c (parameter (ax+b `(10 10) 0 4)))

	(a1 (parameter (ax+b `(10 10) 0 3)))
	(b1 (parameter (ax+b `(10 10) 0 2)))
	(c1 (parameter (ax+b `(10 10) 0 4))))

    (multiple-value-bind (ag bg cg)
	(normal-proceed a b c)
      (multiple-value-bind (ag1 bg1 cg1)
	  (composed-proceed a1 b1 c1)
	(let ((cp (list `(,ag ,ag1)
			`(,bg ,bg1)
			`(,cg ,cg1)))
	      (ans `(2 3 1)))
	  (and
	   (= (vref (car (nth n cp)) 0)
	      (nth n ans))
	   (= (vref (car (nth n cp)) 11)
	      (nth n ans))
	   (M= (car (nth n cp))
	       (second (nth n cp)))))))))

(defun normal-proceed (a b c)
  ;;(proceed-backward (lazy-print (!mul a b)))
  ;; SumBackward bug
  ;; Side effect bug...
  (proceed-backward (!sum (!add c (!mul a b))))
  (values (grad a) (grad b) (grad c)))

(defun composed-proceed (a b c)
  (let* ((k (proceed (!mul a b)))
 	 (out (!add k c)))
    (proceed-backward (!sum out))
    (values (grad a) (grad b) (grad c))))

(deftest proceed-differentiable-p
  (testing "Compute gradients with a proceed function."
    (ok (proceed-continue-test 0))
    (ok (proceed-continue-test 1))
    (ok (proceed-continue-test 2))))

(deftest proceed-composed-test
  (testing "Compute gradients with composed proceed node (e.g.: Proceed(sin(Proceed(x))).grad"
    (ok (<
	 (abs
	  (-
	   (tensor-vec (Proceed (->scal (!Sum (Proceed (cl-waffe2/nn::!Softmax (randn `(3 3))))))))
	   3.0))
	 0.000001))))

(deftest permute-test
  (testing "Compose(LazyReshape, LazyPermute) without creating an additional allocation"
    (ok (equal `(5 1 5) (shape (proceed (!t (!reshape (ax+b `(5 5) 1 0) 5 5 1))))))
    (ok
     (let ((a (parameter (ax+b `(3 4 5) 1 0))))
       (proceed-backward
	(!sum (!permute (!reshape a 10 3 2 1) 0 3 2 1)))
       (grad a)))
    (ok
     (let ((a (parameter (ax+b `(3 4 5) 1 0))))
       (proceed-backward
	(!sum (!permute a 1 0 2)))
       (grad a))))
  (testing "Testing LazyPermute(Copy(X)). Here, assure Copy is not pruned to modify the memory layout."
    (ok (progn
	  (let ((a (parameter (ax+b `(3 4 5) 1 0))))
	    (equal (shape (proceed (!permute a 1 0 2))) `(4 5 3))))))
  (testing "Computes gradients with multiple complicated lazy-permute composed."
    (ok (progn
	  (let ((a (parameter (ax+b `(3 4 5) 1 0))))
	    (proceed-backward
	     (!permute (!sin (!permute a (torch-order 2 1 0))) 0 1 2))
	    (grad a))))
    (ok (every #'=
	       (let ((a (parameter (ax+b `(3 3) 1 0))))
		 (proceed-backward
		  ;;           v
		  (!mul (!t (!add 1.0 a)) (ax+b `(3 3) 1 0)))
		 (tensor-vec (grad a)))
	       (let ((a (parameter (ax+b `(3 3) 1 0))))
		 (proceed-backward
		  ;;      v
		  (!mul (!t a) (ax+b `(3 3) 1 0)))
		 (tensor-vec (grad a)))))))


(deftest log1p-op-fusion-test
  (testing "Testing Log1p simplifier using compiler-macro."
    (ok
     (let ((f (compile nil `(lambda (x) (cl-waffe2/vm:compile-forward-and-backward (!loge (!add x 1.0)))))))
       (= (length (funcall f (randn `(3 3)))) 2)))))


(defun view-diff-test1 ()
  (let ((a (parameter (ax+b `(3 3) 0 1))))
    (proceed-backward
     (!add
      (!view a `(0 2) `(0 2))
      (randn `(2 2))))
    (grad a)))

(deftest view-diff-test
  (testing "Differentiate view"
    (ok (every #'=
	       (tensor-vec (view-diff-test1))
	       #(1 1 0 1 1 0 0 0 0)))))

